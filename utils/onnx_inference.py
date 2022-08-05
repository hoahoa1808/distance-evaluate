import onnxruntime as rt
import cv2
import numpy as np
import h5py


class HDF5DatasetWriter:
    def __init__(self, length, outputPath, embDim, bufSize=10000):
        if outputPath.is_file():
            raise ValueError("The supplied ‘outputPath‘ already "
            "exists and cannot be overwritten. Manually delete "
            "the file before continuing.", outputPath)
        
        self.db = h5py.File(outputPath, "w")
        self.embs = self.db.create_dataset("embs", (length, embDim), dtype=np.float32)
        self.ids = self.db.create_dataset("ids", (length,), dtype="int")

        self.bufSize = bufSize
        self.buffer = {"embs": [], "ids": []}
        self.idx = 0
    
    def add(self, embs, ids):
        self.buffer["embs"].extend(embs)
        self.buffer["ids"].extend(ids)
        if len(self.buffer["embs"]) >= self.bufSize:
            self.flush()
    
    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["embs"])
        self.embs[self.idx:i] = self.buffer["embs"]
        self.ids[self.idx:i] = self.buffer["ids"]
        self.idx = i
        self.buffer = {"embs": [], "ids": []}
    
    def close(self):
        if len(self.buffer["embs"]) > 0:
            self.flush()
        self.db.close()


class OnnxInfer:
    def __init__(self, weight_paths, use_gpu=True):
        if use_gpu:
            self.ort_session = rt.InferenceSession(weight_paths, providers=["CUDAExecutionProvider"])
        else:
            self.ort_session = rt.InferenceSession(weight_paths, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
    
    
    def __call__(self, image_tensor):
        '''
        image: preprocessed image
        '''
        onnx_output = self.ort_session.run(None, {self.input_name: image_tensor})
        return onnx_output
    
    
    # def preprocess(self, img):
    #     '''
    #     image: pil data
    #     return: data preprocessed
    #     '''

    #     input_mean = np.asarray([0.485, 0.456, 0.406])
    #     input_std = np.asarray([0.229, 0.224, 0.225])
    #     img = np.array(img).astype(np.uint8)
    #     img = cv2.resize(img, (128, 256))
    #     img = img.astype(np.float32) / 255.
    #     img = (img - input_mean) / input_std
    #     # img = img[...,::-1] # BGR to RGB
    #     img = img.transpose(2, 0, 1)
    #     img = img.astype(np.float32)
        
    #     return img
        

if __name__ == "__main__":
    net = OnnxInfer(weight_paths="/research/classification/face.evolve/projects/face_mask/weights/webface_batch.onnx")
    
    import timeit
    while True:
        img = np.random.randn(64, 3, 112, 112).astype(np.float32)
        t0 = timeit.default_timer()
        net(img)
        t1 = timeit.default_timer()
        print(t1-t0)
        
        
        
    