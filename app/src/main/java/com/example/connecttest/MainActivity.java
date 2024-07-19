package com.example.connecttest;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import android.Manifest;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_REQUEST_CODE = 1;
    private static final int PERMISSION_REQUEST_CODE = 100;
    private Interpreter tflite;
    private static final int IMAGE_SIZE = 384; // Adjust IMAGE_SIZE as needed for your model
    private static final int OUTPUT_SIZE = 1024; // Adjust OUTPUT_SIZE as needed for your model

    // Define generatedTextView variable
    private TextView generatedTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
            Log.e("TFLite", "Error loading model: ", e);
        }

        // Check for storage permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
        }

        Button selectImageButton = findViewById(R.id.select_image_button);
        generatedTextView = findViewById(R.id.generated_text);

        selectImageButton.setOnClickListener(v -> openImageChooser());
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("blip_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void openImageChooser() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == IMAGE_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                ByteBuffer inputData = preprocessImage(bitmap);
                float[] outputData = runInference(inputData);
                String generatedText = convertOutputToText(outputData);
                runOnUiThread(() -> generatedTextView.setText(generatedText));
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        ByteBuffer imgData = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
        imgData.order(ByteOrder.nativeOrder());

        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat(((val >> 16) & 0xFF) / 255.0f);
                imgData.putFloat(((val >> 8) & 0xFF) / 255.0f);
                imgData.putFloat((val & 0xFF) / 255.0f);
            }
        }

        return imgData;
    }

    private float[] runInference(ByteBuffer inputData) {
        float[] outputData = new float[OUTPUT_SIZE]; // Adjust OUTPUT_SIZE as needed
        tflite.run(inputData, outputData);
        return outputData;
    }

    private String convertOutputToText(float[] outputData) {
        // Convert the model output to text.
        // This is highly dependent on your specific model and use case.
        // A simple placeholder implementation might be:
        StringBuilder text = new StringBuilder();
        for (float value : outputData) {
            text.append((char) value); // Placeholder conversion
        }
        return text.toString();
    }
}