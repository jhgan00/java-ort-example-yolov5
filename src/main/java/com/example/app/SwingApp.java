package com.example.app;

import ai.onnxruntime.OrtException;
import com.example.yolov5.Detection;
import com.example.yolov5.ImageUtil;
import com.example.yolov5.YoloV5;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.List;
import java.util.Objects;


public class SwingApp extends JFrame implements ActionListener {

    private final JMenuItem openItem;
    private final JLabel label;
    private YoloV5 inferenceSession;

    private final int MAX_SIZE = 720;

    private ImageIcon imageIcon;

    public SwingApp() {

        try {
            String modelPath = Objects.requireNonNull(SwingApp.class.getResource("/yolov5s.onnx")).getFile();
            String labelPath = Objects.requireNonNull(SwingApp.class.getResource("/coco.names")).getFile();
            inferenceSession = new YoloV5(modelPath, labelPath, 0.25f, 0.45f, -1);
        }
        catch (OrtException | IOException exception) {
            exception.printStackTrace();
            System.exit(1);
        }

        setTitle("Yolov5 Demo");
        setSize(640, 640);
        JMenuBar mbar = new JMenuBar();
        JMenu m = new JMenu("File");
        openItem = new JMenuItem("Open");
        openItem.addActionListener(this);
        m.add(openItem);
        mbar.add(m);
        setJMenuBar(mbar);
        label = new JLabel();
        Container contentPane = getContentPane();
        contentPane.add(label, "Center");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);
        setResizable(true);

    }


    public static void main(String[] args) {
        new SwingApp();
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Object source = e.getSource();

        if (source == openItem) {

            JFileChooser chooser = new JFileChooser();
            chooser.setFileFilter(new FileNameExtensionFilter(
                    "Image files",
                    "png", "jpg", "jpeg")
            );

            int r = chooser.showOpenDialog(this);

            if (r == JFileChooser.APPROVE_OPTION) {
                String name = chooser.getSelectedFile().getPath();

                // open image file
                Mat img = Imgcodecs.imread(name);
                if (Math.max(img.width(), img.height()) > MAX_SIZE) {
                    ImageUtil.resizeWithPadding(img, img, MAX_SIZE, MAX_SIZE);
                }

                // run detection
                try {
                    List<Detection> detectionList = inferenceSession.run(img);
                    ImageUtil.drawPredictions(img, detectionList);
                }
                catch (OrtException ortException) {
                    ortException.printStackTrace();
                }

                // Displaying in the GUI
                MatOfByte matOfByte = new MatOfByte();
                Imgcodecs.imencode(".png", img, matOfByte);

                imageIcon = new ImageIcon(matOfByte.toArray());
                imageIcon.getImage().flush();
                label.setIcon(imageIcon);
                pack();

            }
        }
    }
}
