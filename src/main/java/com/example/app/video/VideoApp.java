package com.example.app.video;

import ai.onnxruntime.OrtException;
import com.example.yolo.Detection;
import com.example.yolo.ImageUtil;
import com.example.yolo.ModelFactory;
import com.example.yolo.Yolo;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import java.io.IOException;
import java.util.List;


public class VideoApp  {

    private Yolo inferenceSession;

    public VideoApp() {
        ModelFactory modelFactory = new ModelFactory();
        try {
            this.inferenceSession = modelFactory.getModel("model.properties");
        } catch (OrtException | IOException exception) {
            exception.printStackTrace();
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        nu.pattern.OpenCV.loadLocally();
        VideoApp app = new VideoApp();

        VideoCapture cap = new VideoCapture(0);
        cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 800);

        JFrame jframe = new JFrame("Title");
        jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel vidpanel = new JLabel();
        jframe.setContentPane(vidpanel);
        jframe.setVisible(true);
        jframe.setResizable(true);
        jframe.setSize(1280, 800);

        if (!cap.isOpened()) {
            System.exit(1);
        }

        Mat frame = new Mat();
        MatOfByte matofByte = new MatOfByte();

        while ( cap.read(frame) ) {

            ImageUtil.resizeWithPadding(frame, frame, 1280, 800);

            // run detection
            try {
                List<Detection> detectionList = app.inferenceSession.run(frame);
                ImageUtil.drawPredictions(frame, detectionList);
            } catch (OrtException ortException) {
                ortException.printStackTrace();
            }
            Imgcodecs.imencode(".jpg", frame, matofByte);
            ImageIcon imageIcon = new ImageIcon(matofByte.toArray());
            vidpanel.setIcon(imageIcon);
            vidpanel.repaint();
        }

        cap.release();
    }

}
