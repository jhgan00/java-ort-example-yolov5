package com.example.yolo;

import ai.onnxruntime.OrtException;
import com.example.app.ConfigReader;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

public class ModelFactory {


    public Yolo getModel(String propertiesFilePath) throws IOException, OrtException, NotImplementedException {

        ConfigReader configReader = new ConfigReader();
        Properties properties = configReader.readProperties(propertiesFilePath);

        String modelName = properties.getProperty("modelName");
        File file = ResourceUtils.getFile("classpath:" + properties.getProperty("modelPath"));
        File file2 = ResourceUtils.getFile("classpath:coco.names");
        String modelPath =  String.valueOf((file));
        String labelPath =  String.valueOf((file2));
        float confThreshold = Float.parseFloat(properties.getProperty("confThreshold"));
        float nmsThreshold = Float.parseFloat(properties.getProperty("nmsThreshold"));
        int gpuDeviceId = Integer.parseInt(properties.getProperty("gpuDeviceId"));

        if (modelName.equalsIgnoreCase("yolov5")) {
            return new YoloV5(modelPath, labelPath, confThreshold, nmsThreshold, gpuDeviceId);
        }
        else if (modelName.equalsIgnoreCase("yolov8")) {
            return new YoloV8(modelPath, labelPath, confThreshold, nmsThreshold, gpuDeviceId);
        }
        else {
            throw new NotImplementedException();
        }

    }
}
