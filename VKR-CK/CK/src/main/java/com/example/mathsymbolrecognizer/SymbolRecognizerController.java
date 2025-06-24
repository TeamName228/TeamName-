package com.example.mathsymbolrecognizer;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Map;

@Controller
public class SymbolRecognizerController {

    @GetMapping("/")
    public String index(Model model) {
        return "index";
    }

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("imageData") String imageData, Model model) {
        if (imageData == null || imageData.isEmpty()) {
            model.addAttribute("error", "Изображение не обрезано или не загружено!");
            return "index";
        }

        try {
            // Декодируем dataURL
            String base64Image = imageData.split(",")[1];
            byte[] imageBytes = Base64.getDecoder().decode(base64Image);
            BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));

            // Сохраняем обрезанное изображение
            String uploadDir = "Uploads/";
            File dir = new File(uploadDir);
            if (!dir.exists()) dir.mkdirs();
            String fileName = System.currentTimeMillis() + "_cropped.jpg";
            Path filePath = Path.of(uploadDir, fileName);
            ImageIO.write(image, "jpg", filePath.toFile());

            // Вызываем Python-скрипт
            ProcessBuilder pb = new ProcessBuilder("python", "python/recognize_symbol.py", filePath.toString());
            pb.redirectErrorStream(false); // Отдельные потоки для stdout и stderr
            Process process = pb.start();
            String result = new String(process.getInputStream().readAllBytes(), "UTF-8").trim();
            int exitCode = process.waitFor();

            Files.deleteIfExists(filePath);

            if (exitCode == 0 && !result.isEmpty() && !result.startsWith("Ошибка")) {
                // Парсим JSON
                ObjectMapper objectMapper = new ObjectMapper();
                Map<String, String> resultMap = objectMapper.readValue(result, Map.class);
                model.addAttribute("symbol", resultMap.get("symbol"));
                model.addAttribute("latex", resultMap.get("latex"));
                model.addAttribute("confidence", resultMap.get("confidence"));
                model.addAttribute("success", "Символ успешно распознан!");
            } else {
                String error = new String(process.getErrorStream().readAllBytes(), "UTF-8").trim();
                model.addAttribute("error", result.isEmpty() ? "Ошибка распознавания!" : (error.isEmpty() ? result : error));
            }

        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            model.addAttribute("error", "Ошибка парсинга JSON: " + e.getMessage());
        } catch (IOException | InterruptedException e) {
            model.addAttribute("error", "Ошибка обработки: " + e.getMessage());
        }

        return "index";
    }
}