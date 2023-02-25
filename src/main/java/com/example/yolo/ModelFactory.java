package com.example.yolo;

import ai.onnxruntime.OrtException;
import com.example.app.ConfigReader;

import java.io.IOException;
import java.util.Objects;
import java.util.Properties;

public class ModelFactory {


    public Yolo getModel(String propertiesFilePath) throws IOException, OrtException, NotImplementedException {

        ConfigReader configReader = new ConfigReader();
        Properties properties = configReader.readProperties(propertiesFilePath);

        String modelName = properties.getProperty("modelName");
        String modelPath = Objects.requireNonNull(getClass().getClassLoader().getResource(properties.getProperty("modelPath"))).getFile();
        String labelPath = Objects.requireNonNull(getClass().getClassLoader().getResource(properties.getProperty("labelPath"))).getFile();
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
