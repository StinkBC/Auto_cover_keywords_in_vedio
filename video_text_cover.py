import os
import cv2
import pytesseract
from PIL import Image
import numpy as np
import time
import threading
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class VideoKeywordProcessor:
    def __init__(self, config):
        """初始化视频处理器（三级检测机制）"""
        self.input_dir = os.path.abspath(config['INPUT_DIRECTORY'])
        self.output_dir = os.path.abspath(config['OUTPUT_DIRECTORY'])
        self.keywords = [kw.strip().lower() for kw in config['KEYWORDS'].split(',') if kw.strip()]
        self.processed_log = config.get('PROCESSED_LOG', "processed_files.txt")
        self.primary_interval = int(config['PRIMARY_DETECTION_INTERVAL'])  # 1000帧
        self.secondary_interval = int(config['SECONDARY_DETECTION_INTERVAL'])  # 100帧
        self.max_workers = int(config['MAX_WORKERS'])
        self.debug = config.getboolean('DEBUG', False)
        
        # 初始化已处理文件列表
        self.processed_files = self.load_processed_files()
        
        # 配置Tesseract路径
        if 'TESSERACT_PATH' in config and config['TESSERACT_PATH']:
            pytesseract.pytesseract.tesseract_cmd = config['TESSERACT_PATH']
        
        # 线程锁
        self.log_lock = threading.Lock()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        
        # GPU加速检测
        self.use_gpu = self.check_gpu_availability()
        self.log(f"GPU加速: {'已启用' if self.use_gpu else '未启用，使用CPU'}")

    def log(self, message):
        """线程安全的日志输出"""
        with self.log_lock:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def debug_log(self, message):
        """调试日志"""
        if self.debug:
            self.log(f"DEBUG: {message}")

    def check_gpu_availability(self):
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False

    def load_processed_files(self):
        if os.path.exists(self.processed_log):
            with open(self.processed_log, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        return []

    def add_processed_file(self, file_path):
        with self.log_lock:
            with open(self.processed_log, 'a', encoding='utf-8') as f:
                f.write(f"{file_path}\n")
            self.processed_files.append(file_path)

    def is_processed(self, file_path):
        return file_path in self.processed_files

    def get_output_path(self, input_path):
        relative_path = os.path.relpath(input_path, self.input_dir)
        output_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path

    def preprocess_frame(self, frame):
        """图像预处理增强OCR识别率"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if not self.use_gpu else frame
        
        # 去噪处理
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def detect_keywords(self, frame):
        """检测帧中关键字并返回区域"""
        # 预处理帧
        processed = self.preprocess_frame(frame)
        
        # 转换为PIL图像
        pil_img = Image.fromarray(processed)
        
        # OCR识别
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['text'])
        
        regions = []
        for i in range(n_boxes):
            # 降低置信度阈值，提高识别率
            if int(data['conf'][i]) > 30:  # 从60降至30
                text = data['text'][i].lower().strip()
                if text and any(kw in text for kw in self.keywords):
                    x, y, w, h = (data['left'][i], data['top'][i], 
                                 data['width'][i], data['height'][i])
                    # 扩大区域确保完全覆盖
                    x1 = max(0, x - 2)
                    y1 = max(0, y - 2)
                    x2 = min(frame.shape[1], x + w + 2)
                    y2 = min(frame.shape[0], y + h + 2)
                    regions.append((x1, y1, x2, y2))
                    self.debug_log(f"检测到关键字: '{text}' 在区域: ({x1},{y1})-({x2},{y2}) 置信度: {data['conf'][i]}")
        
        return regions

    def blur_regions(self, frame, regions):
        """用黑色方块覆盖指定区域"""
        # 创建帧副本避免修改原图
        frame_copy = frame.copy()
        
        for (x1, y1, x2, y2) in regions:
            # 确保坐标有效
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # 绘制黑色填充矩形
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)  # -1表示填充
        
        return frame_copy

    def process_video_segment(self, cap, start_frame, end_frame):
        """处理视频片段，返回每个帧的处理结果"""
        results = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            regions = self.detect_keywords(frame)
            processed_frame = self.blur_regions(frame, regions)
            results.append((frame_num, processed_frame))
            
            # 调试输出
            if frame_num % 50 == 0:
                self.debug_log(f"处理片段帧: {frame_num}/{end_frame}")
                
        return results

    def find_transition_point(self, cap, start, end):
        """在区间内用二级间隔(100帧)检测状态变化点"""
        self.debug_log(f"开始二级检测: 帧 {start} 至 {end}")
        
        # 记录每个检测点的状态
        check_points = []
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # 按二级间隔检测
        for frame_num in range(start, end + 1, self.secondary_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
                
            regions = self.detect_keywords(frame)
            has_keyword = len(regions) > 0
            check_points.append((frame_num, has_keyword))
            self.debug_log(f"二级检测点 {frame_num}: {'含关键字' if has_keyword else '无关键字'}")
        
        # 恢复位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        # 寻找状态变化的区间
        transition_intervals = []
        for i in range(1, len(check_points)):
            prev_frame, prev_state = check_points[i-1]
            curr_frame, curr_state = check_points[i]
            
            if prev_state != curr_state:
                # 发现状态变化，记录这个子区间
                transition_intervals.append((prev_frame, curr_frame))
        
        return transition_intervals

    def process_video(self, input_path):
        """处理单个视频文件（三级检测机制）"""
        try:
            start_time = time.time()
            self.log(f"开始处理视频: {input_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.log(f"无法打开视频: {input_path}")
                return False
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_path = self.get_output_path(input_path)
            
            # 编码器选择（避开OpenH264）
            output_ext = os.path.splitext(output_path)[1].lower()
            fourcc_map = {
                ('.mp4', '.m4v'): 'mp4v',  # 使用mp4v替代avc1，避免依赖OpenH264
                ('.mov', '.qt'): 'mp4v',
                ('.avi',): 'XVID',
                ('.mkv',): 'VP09',
                ('.flv',): 'FLV1',
                ('.wmv',): 'WMV2'
            }
            fourcc_code = 'mp4v'
            for exts, code in fourcc_map.items():
                if output_ext in exts:
                    fourcc_code = code
                    break
            # 不支持的格式转为AVI
            if output_ext not in [ext for exts in fourcc_map.keys() for ext in exts]:
                output_path = os.path.splitext(output_path)[0] + '.avi'
                fourcc_code = 'XVID'
                self.log(f"转换不支持的格式为AVI: {output_path}")
            
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 初始化状态变量
            frame_count = 0
            last_detection_frame = 0
            last_regions = []
            last_has_keyword = False
            
            # 处理第一帧
            ret, first_frame = cap.read()
            if not ret:
                self.log(f"视频为空: {input_path}")
                return False
                
            first_regions = self.detect_keywords(first_frame)
            last_has_keyword = len(first_regions) > 0
            last_regions = first_regions
            out.write(self.blur_regions(first_frame, first_regions))
            frame_count = 1
            
            # 主处理循环
            while frame_count < total_frames:
                # 检查是否达到一级检测间隔
                if frame_count - last_detection_frame >= self.primary_interval:
                    # 记录当前位置
                    current_pos = frame_count
                    
                    # 在间隔终点进行检测
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, check_frame = cap.read()
                    if not ret:
                        break
                        
                    current_regions = self.detect_keywords(check_frame)
                    current_has_keyword = len(current_regions) > 0
                    
                    # 检查状态是否变化
                    if current_has_keyword != last_has_keyword:
                        self.log(f"检测到状态变化，开始二级检测: 帧 {last_detection_frame} 至 {frame_count}")
                        
                        # 用二级间隔(100帧)寻找变化区间
                        transition_intervals = self.find_transition_point(
                            cap, last_detection_frame, frame_count)
                        
                        if transition_intervals:
                            # 对每个变化区间进行逐帧处理
                            for (start, end) in transition_intervals:
                                self.log(f"在区间 {start}-{end} 发现状态变化，开始逐帧处理")
                                
                                # 处理这个区间的所有帧
                                segment_results = self.process_video_segment(cap, start, end)
                                for frame_num, processed_frame in segment_results:
                                    # 确保帧序号正确
                                    if frame_num >= last_detection_frame and frame_num <= frame_count:
                                        out.write(processed_frame)
                        else:
                            # 未找到具体变化点，用当前状态处理整个区间
                            self.debug_log(f"未找到精确变化点，用当前状态处理区间 {last_detection_frame}-{frame_count}")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, last_detection_frame + 1)
                            
                            for _ in range(last_detection_frame + 1, frame_count + 1):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                processed = self.blur_regions(frame, current_regions)
                                out.write(processed)
                        
                        # 更新状态
                        last_has_keyword = current_has_keyword
                        last_regions = current_regions
                        last_detection_frame = frame_count
                    else:
                        # 状态未变化，用上次的区域处理中间帧
                        self.debug_log(f"状态未变化，批量处理帧 {last_detection_frame+1} 至 {frame_count}")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, last_detection_frame + 1)
                        
                        for _ in range(last_detection_frame + 1, frame_count + 1):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            processed = self.blur_regions(frame, last_regions)
                            out.write(processed)
                        
                        last_detection_frame = frame_count
                
                # 处理下一帧
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # 进度更新
                if frame_count % 500 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = int(time.time() - start_time)
                    self.log(f"处理中: {input_path} {progress:.1f}% 已用: {elapsed}s")
        
        except Exception as e:
            self.log(f"处理出错 {input_path}: {str(e)}")
            return False
        finally:
            try:
                cap.release()
                out.release()
            except:
                pass
        
        duration = int(time.time() - start_time)
        self.log(f"处理完成: {input_path}，耗时: {duration}秒")
        return True

    def process_directory(self):
        """多线程处理目录下所有视频"""
        start_time = time.time()
        self.log(f"开始处理目录: {self.input_dir}")
        self.log(f"关键字列表: {', '.join(self.keywords)}")
        self.log(f"检测间隔: 一级={self.primary_interval}帧, 二级={self.secondary_interval}帧, 线程数={self.max_workers}")
        
        # 收集所有待处理视频
        video_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(self.video_extensions):
                    input_path = os.path.join(root, file)
                    if not self.is_processed(input_path):
                        video_files.append(input_path)
                    else:
                        self.log(f"已处理，跳过: {input_path}")
        
        total_videos = len(video_files)
        processed_count = 0
        self.log(f"发现{total_videos}个待处理视频，开始并行处理...")
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_video, path): path for path in video_files}
            
            for future in as_completed(futures):
                path = futures[future]
                if future.result():
                    self.add_processed_file(path)
                    processed_count += 1
                self.log(f"整体进度: {processed_count}/{total_videos} 个视频")
        
        total_duration = int(time.time() - start_time)
        self.log(f"全部处理完成！共处理{processed_count}/{total_videos}个视频，总耗时: {total_duration}秒")

def load_config(config_path="config.ini"):
    """加载配置文件，不存在则创建默认配置"""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_path):
        # 创建默认配置
        config['General'] = {
            'INPUT_DIRECTORY': 'input_videos',
            'OUTPUT_DIRECTORY': 'output_videos',
            'KEYWORDS': '敏感词1, 敏感词2, confidential',
            'PRIMARY_DETECTION_INTERVAL': '1000',
            'SECONDARY_DETECTION_INTERVAL': '100',
            'MAX_WORKERS': '4',
            'PROCESSED_LOG': 'processed_files.txt',
            'TESSERACT_PATH': '',
            'DEBUG': 'False'
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            config.write(f)
        print(f"已创建默认配置文件: {config_path}")
    
    config.read(config_path, encoding='utf-8')
    return config['General']

if __name__ == "__main__":
    # 加载配置
    config = load_config()
    
    # 创建处理器并开始处理
    processor = VideoKeywordProcessor(config)
    processor.process_directory()
    