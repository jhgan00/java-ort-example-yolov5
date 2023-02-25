package com.example.app.swing;

import ai.onnxruntime.OrtException;
import com.example.yolo.Detection;
import com.example.yolo.ImageUtil;
import com.example.yolo.ModelFactory;
import com.example.yolo.Yolo;
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


public class SwingApp extends JFrame implements ActionListener {

    private final JMenuItem openItem;
    private final JLabel label;
    private Yolo inferenceSession;

    public SwingApp() {

        ModelFactory modelFactory = new ModelFactory();

        try {
            this.inferenceSession = modelFactory.getModel("model.properties");
        } catch (OrtException | IOException exception) {
            exception.printStackTrace();
            System.exit(1);
        }

        setTitle("Yolo Demo");
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
                int MAX_SIZE = 720;
                if (Math.max(img.width(), img.height()) > MAX_SIZE) {
                    ImageUtil.resizeWithPadding(img, img, MAX_SIZE, MAX_SIZE);
                }

                // run detection
                try {
                    List<Detection> detectionList = inferenceSession.run(img);
                    ImageUtil.drawPredictions(img, detectionList);
                } catch (OrtException ortException) {
                    ortException.printStackTrace();
                }

                // Displaying in the GUI
                MatOfByte matOfByte = new MatOfByte();
                Imgcodecs.imencode(".png", img, matOfByte);

                ImageIcon imageIcon = new ImageIcon(matOfByte.toArray());
                imageIcon.getImage().flush();
                label.setIcon(imageIcon);
                pack();

            }
        }
    }
}
