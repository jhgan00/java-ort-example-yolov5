package com.example.yolov5;

import ai.onnxruntime.OrtException;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Objects;
import com.google.gson.Gson;

public class CLI_App {

    private YoloV5 inferenceSession;
    private Gson gson;
    public CLI_App() {
        try {

            String modelPath = Objects.requireNonNull(SwingApp.class.getResource("/yolov5s.onnx")).getFile();
            String labelPath = Objects.requireNonNull(SwingApp.class.getResource("/coco.names")).getFile();
            this.inferenceSession = new YoloV5(modelPath, labelPath, 0.25f, 0.45f, -1);
            this.gson = new Gson();
        }
        catch (OrtException | IOException exception) {
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

            if ( "q".equals(input) | "Q".equals(input) ) {
                System.out.println("Exit");
                System.exit(0);
            }

            File f = new File(input);
            if ( !f.exists() ) {
                System.out.println("File does not exists: " + input);
                continue;
            }
            if ( f.isDirectory() ) {
                System.out.println(input + " is a directory");
                continue;
            }

            Mat img = Imgcodecs.imread(input, Imgcodecs.IMREAD_COLOR);
            if (img.dataAddr() == 0 ) {
                System.out.println("Could not open image: " + input);
                continue;
            }
            // run detection
            try {
                List<Detection> detectionList = app.inferenceSession.run(img);
                ImageUtil.drawPredictions(img, detectionList);
                System.out.println(app.gson.toJson(detectionList));
                Imgcodecs.imwrite("predictions.jpg", img);
            }
            catch (OrtException | IOException ortException) {
                ortException.printStackTrace();
            }

        }

    }

}
