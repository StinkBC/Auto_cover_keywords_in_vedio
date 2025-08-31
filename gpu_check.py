import cv2
print(cv2.getBuildInformation())


try:
    # 检查CUDA是否可用
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA可用设备数: {cv2.cuda.getCudaEnabledDeviceCount()}")
    print(f"GPU加速是否可用: {'是' if cuda_available else '否'}")
    
    # 测试简单的GPU操作
    if cuda_available:
        # 创建一个测试矩阵并上传到GPU
        mat = cv2.Mat(100, 100, cv2.CV_8UC3)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(mat)
        print("GPU内存上传成功，加速加速功能正常")
except Exception as e:
    print(f"GPU加速测试失败: {str(e)}")