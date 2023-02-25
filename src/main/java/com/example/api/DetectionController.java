package com.example.api;

import ai.onnxruntime.OrtException;
import com.example.yolo.Detection;
import com.example.yolo.YoloV5;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;


@RestController
public class DetectionController {
    static private final List<String> mimeTypes = Arrays.asList("image/png", "image/jpeg");
    private final YoloV5 inferenceSession;
    private final Logger LOGGER = LoggerFactory.getLogger(DetectionController.class);

    private DetectionController() throws OrtException, IOException {
        String modelPath = Objects.requireNonNull(DetectionController.class.getResource("/yolov5s.onnx")).getFile();
        String labelPath = Objects.requireNonNull(DetectionController.class.getResource("/coco.names")).getFile();
        this.inferenceSession = new YoloV5(modelPath, labelPath, 0.25f, 0.45f, -1);
    }

    @PostMapping(value = "/detection", consumes = {"multipart/form-data"}, produces = {"application/json"})
    public List<Detection> detection(MultipartFile uploadFile) throws OrtException, IOException {
        if (!mimeTypes.contains(uploadFile.getContentType())) throw new UploadFileException(ErrorCode.INVALID_MIME_TYPE);
        byte[] bytes = uploadFile.getBytes();
        Mat img = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_COLOR);
        List<Detection> result = inferenceSession.run(img);
        LOGGER.info("POST 200");
        return result;
    }
}
