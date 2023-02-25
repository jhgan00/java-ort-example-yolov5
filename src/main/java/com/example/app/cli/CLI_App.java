package com.example.app.cli;

import ai.onnxruntime.OrtException;
import com.example.yolo.*;
import com.google.gson.Gson;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.util.List;

public class CLI_App {

    private Yolo inferenceSession;
    private Gson gson;

    public CLI_App() {

        ModelFactory modelFactory = new ModelFactory();

        try {
            this.inferenceSession = modelFactory.getModel("./model.properties");
            this.gson = new Gson();
        } catch (OrtException | IOException exception) {
            exception.printStackTrace();
            System.exit(1);
        }
    }

    public static void main(String[] args) {

        CLI_App app = new CLI_App();

        while (true) {

            System.out.print("Enter image path (enter 'q' or 'Q' to exit): ");

            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            String input = null;

            try {
                input = br.readLine();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }

            if ("q".equals(input) | "Q".equals(input)) {
                System.out.println("Exit");
                System.exit(0);
            }

            File f = new File(input);
            if (!f.exists()) {
                System.out.println("File does not exists: " + input);
                continue;
            }
            if (f.isDirectory()) {
                System.out.println(input + " is a directory");
                continue;
            }

            Mat img = Imgcodecs.imread(input, Imgcodecs.IMREAD_COLOR);
            if (img.dataAddr() == 0) {
                System.out.println("Could not open image: " + input);
                continue;
            }
            // run detection
            try {
                List<Detection> detectionList = app.inferenceSession.run(img);
                ImageUtil.drawPredictions(img, detectionList);
                System.out.println(app.gson.toJson(detectionList));
                Imgcodecs.imwrite("predictions.jpg", img);
            } catch (OrtException ortException) {
                ortException.printStackTrace();
            }

        }

    }

}
