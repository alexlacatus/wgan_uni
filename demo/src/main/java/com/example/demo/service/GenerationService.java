package com.example.demo.service;// src/main/java/.../service/GenerationService.java

import com.example.demo.helper.TransactionMapper;
import com.example.demo.helper.TransactionToSyntheticMapper;
import com.example.demo.model.GeneratorRequestDTO;
import com.example.demo.model.SyntheticTransaction;
import com.example.demo.model.TransactionDTO;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.util.Pair;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Service
public class GenerationService {

    private final RestTemplate restTemplate;
    private final TransactionMapper transactionMapper;
    private final TransactionToSyntheticMapper transactionToSyntheticMapper;
    private final CsvFileService csvFileService;

    @Value("${gan.api.url}")
    private String ganApiUrl;

    public GenerationService(RestTemplate restTemplate, TransactionMapper transactionMapper, TransactionToSyntheticMapper transactionToSyntheticMapper, CsvFileService csvFileService) {
        this.restTemplate = restTemplate;
        this.transactionMapper = transactionMapper;
        this.transactionToSyntheticMapper = transactionToSyntheticMapper;
        this.csvFileService = csvFileService;
    }

    public String generateTransactions(GeneratorRequestDTO requestDTO) throws IOException {
        String url = ganApiUrl + "/generate";

        // Cererea POST către API-ul Python
        Map<String, Integer> requestBody = Collections.singletonMap("num_samples", requestDTO.getSize());

        SyntheticTransaction[] transactions = restTemplate.postForObject(
                url,
                requestBody,
                SyntheticTransaction[].class
        );

        if(transactions == null){
            throw new RuntimeException("No transactions found, error with the generator");
        }

        List<TransactionDTO> finalTransactions = transactionMapper.mapToDTO(Arrays.stream(transactions).toList(),requestDTO.getFeatureDefinitions(),requestDTO.getProductNames());

        return csvFileService.writeTransactionsToCsv(finalTransactions);
    }

    public String reverseMapAndWriteRawData(
            String inputFilename,
            List<Pair<String, List<String>>> featureDefinitions,
            List<String> productNames) throws IOException {

        // 1. Read DTOs from the input CSV file
        // NOTE: You need a CsvFileService.readCsvToDtos(filename, featureDefs) method
        // which handles CSV parsing into a List<TransactionDTO>. (This method is assumed for now).
        List<TransactionDTO> dtos = csvFileService.readCsvToDtos(inputFilename, featureDefinitions.size());

        if (dtos.isEmpty()) {
            return null;
        }

        // 2. Reverse Map DTOs to SyntheticTransaction format
        // This is complex! We need a new mapper method (assumed to exist in TransactionMapper)
        // that converts category names back into a numerical format, typically the center of the bin.
        List<SyntheticTransaction> rawTransactions = transactionToSyntheticMapper.mapToSynthetic (dtos, featureDefinitions, productNames);

        // 3. Write raw SyntheticTransaction data to a new CSV file
        // NOTE: We need a CsvFileService.writeSyntheticTransactionsToCsv() method.
        // This new method will write the raw numeric/ID data suitable for loading into PyTorch.
        String outputFilename = "raw_mapped_" + inputFilename;

        return csvFileService.writeSyntheticTransactionsToCsv(rawTransactions, outputFilename);
    }
}