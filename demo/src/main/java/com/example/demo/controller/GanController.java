package com.example.demo.controller;

import com.example.demo.model.GeneratorRequestDTO;
import com.example.demo.model.RawConversionRequestDTO;
import com.example.demo.model.SyntheticTransaction;
import com.example.demo.service.GenerationService;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.io.IOException;
import java.net.URI;
import java.util.List;

@RestController
@RequestMapping("/api/transactions")
public class GanController {

    private final GenerationService generationService;

    public GanController(GenerationService generationService) {
        this.generationService = generationService;
    }

    // --- POST METHOD 1 (Existing): Generate Data, Write CSV, Return Link ---
    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_PLAIN_VALUE)
    public ResponseEntity<String> generateAndDownloadCsv(@RequestBody GeneratorRequestDTO requestDTO) {

        // 1. Validation (Using accessor appropriate for your DTO, assuming it's size or count)
        int count = requestDTO.getSize();
        if (count < 1 || count > 1000) {
            return ResponseEntity.badRequest().body("Count must be between 1 and 1000.");
        }

        try {
            // 2. Execute generation and persistence, getting the filename back
            String filename = generationService.generateTransactions(requestDTO);

            if (filename == null) {
                return ResponseEntity.status(HttpStatus.NO_CONTENT).body("No transactions generated.");
            }

            // 3. Construct the full URI to the file
            URI fileUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                    .path("/download/").path(filename)
                    .build().toUri();

            return ResponseEntity.status(HttpStatus.CREATED)
                    .body(fileUri.toString());

        } catch (IOException e) {
            System.err.println("Error processing transaction request: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Failed to generate or save CSV file due to I/O error.");
        }
    }

    // --- POST METHOD 2 (New): Convert Categorized Data back to Raw Numerical Format ---
    @PostMapping("/raw-convert")
    public ResponseEntity<String> convertToRaw(
            @RequestBody RawConversionRequestDTO requestDTO) throws IOException {

        // 1. Input Validation


        // 2. Call Service to perform the reverse mapping
        String resultCsv = generationService.reverseMapAndWriteRawData(
                requestDTO.getCsvFile(),
                requestDTO.getFeatureDefinitions(),
                requestDTO.getProductNames()
        );

        // 3. Return the list of raw numerical objects
        return ResponseEntity.ok(resultCsv);
    }
}