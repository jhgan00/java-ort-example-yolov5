# Onnxruntime Java Example : yolov5

![](assets/predictions.jpg)

```bash
mvn clean compile

# GUI App
mvn exec:java -Dexec.mainClass="com.example.yolov5.SwingApp" -Dexec.classpathScope=test

# CLI APP
mvn exec:java -Dexec.mainClass="com.example.yolov5.CLI_App" -Dexec.classpathScope=test
```