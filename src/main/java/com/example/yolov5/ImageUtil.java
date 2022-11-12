package com.example.yolov5;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.List;


public class ImageUtil {

    public static Mat resizeWithPadding(Mat src, int width, int height) {

        Mat dst = new Mat();
        int oldW = src.width();
        int oldH = src.height();

        double r = Math.min((double )width / oldW, (double) height / oldH);

        int newUnpadW = (int) Math.round(oldW * r);
        int newUnpadH = (int) Math.round(oldH * r);

        int dw = (width - newUnpadW) / 2;
        int dh = (height - newUnpadH) / 2;

        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);

        Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
        Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);

        return dst;

    }

    public static void resizeWithPadding(Mat src, Mat dst, int width, int height) {

        int oldW = src.width();
        int oldH = src.height();

        double r = Math.min((double )width / oldW, (double) height / oldH);

        int newUnpadW = (int) Math.round(oldW * r);
        int newUnpadH = (int) Math.round(oldH * r);

        int dw = (width - newUnpadW) / 2;
        int dh = (height - newUnpadH) / 2;

        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);

        Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
        Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);

    }

    public static void whc2cwh (float[] src, float[] dst, int start) {
        int j=start;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                dst[j] = src[i];
                j++;
            }
        }
    }

    public static float[] whc2cwh (float[] src) {
        float[] chw = new float[src.length];
        int j=0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    public static byte[] whc2cwh (byte[] src) {
        byte[] chw = new byte[src.length];
        int j=0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }
    public static void drawPredictions(Mat img, List<Detection> detectionList) {
        // debugging image
        for (Detection detection : detectionList) {

            float[] bbox = detection.bbox();
            Scalar color = new Scalar(249, 218, 60);
            Imgproc.rectangle ( img,                    //Matrix obj of the image
                    new Point(bbox[0], bbox[1]),        //p1
                    new Point(bbox[2], bbox[3]),       //p2
                    color,     //Scalar object for color
                    2                        //Thickness of the line
            );
            Imgproc.putText(
                    img,
                    detection.label(),
                    new Point(bbox[0] - 1, bbox[1] - 5),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    .5, color,
                    1);
        }
    }

}
