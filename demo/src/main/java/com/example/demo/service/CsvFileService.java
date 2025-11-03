package com.example.demo.service;

import com.example.demo.model.TransactionDTO;
import com.example.demo.model.SyntheticTransaction;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class CsvFileService {

    private final static String FILE_STORAGE_DIR = "generated_files";
    private final static String CSV_SEPARATOR = ",";
    private final static String LINE_SEPARATOR = "\n";


    public CsvFileService() {
        try {
            Files.createDirectories(Paths.get(FILE_STORAGE_DIR));
        } catch (IOException e) {
            System.err.println("CRITICAL ERROR: Could not create file storage directory: " + FILE_STORAGE_DIR);
        }
    }

    /**
     * Writes a list of mapped TransactionDTOs (categorized data) to a CSV file.
     * * @param transactions The list of DTOs to write.
     * @return The filename (relative path) to the generated file.
     * @throws IOException if file writing fails.
     */
    public String writeTransactionsToCsv(List<TransactionDTO> transactions) throws IOException {

        if (transactions.isEmpty()) {
            return null;
        }

        String csvContent = convertToCsv(transactions);

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = "transactions_" + timestamp + ".csv";
        Path filePath = Paths.get(FILE_STORAGE_DIR, filename);

        Files.write(filePath, csvContent.getBytes());

        return filename;
    }

    /**
     * Writes a list of raw SyntheticTransaction objects (numeric/ID data) to a CSV file.
     * This is the reverse process, saving data in a format suitable for the GAN/PyTorch.
     * * @param transactions The list of raw SyntheticTransaction objects.
     * @param baseFilename The base name for the output file (e.g., "raw_mapped_input.csv").
     * @return The filename (relative path) to the generated file.
     * @throws IOException if file writing fails.
     */
    public String writeSyntheticTransactionsToCsv(List<SyntheticTransaction> transactions, String baseFilename) throws IOException {

        if (transactions.isEmpty()) {
            return null;
        }

        String csvContent = convertToRawCsv(transactions);

        // Create a unique filename based on the base name
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = baseFilename.replace(".csv", "") + timestamp + ".csv";
        Path filePath = Paths.get(FILE_STORAGE_DIR, filename);

        Files.write(filePath, csvContent.getBytes());

        return filename;
    }

    /**
     * Reads a CSV file and converts its data rows into a list of TransactionDTOs.
     * NOTE: This is a complex operation typically requiring a dedicated CSV library
     * (like OpenCSV or Commons CSV) to correctly parse headers and quoted strings.
     * This implementation provides a simplified placeholder and assumes the caller
     * knows the CSV structure.
     * * @param inputFilename The name of the file to read.
     * @param expectedFeatureCount The expected number of continuous features to parse.
     * @return A list of TransactionDTOs.
     * @throws IOException if file reading fails.
     */
    public List<TransactionDTO> readCsvToDtos(String inputFilename, int expectedFeatureCount) throws IOException {
        Path filePath = Paths.get(FILE_STORAGE_DIR, inputFilename);

        if (!Files.exists(filePath)) {
            throw new IOException("Input file not found: " + inputFilename);
        }

        List<String> allLines = Files.readAllLines(filePath);
        if (allLines.isEmpty()) {
            return new ArrayList<>();
        }

        // 1. Parse Header
        // The header structure is: "quantity,productName,FeatureName1,FeatureName2,..."
        String headerLine = allLines.get(0);

        // Split header: skip "quantity" and "productName"
        String[] headerColumns = headerLine.split(",");

        if (headerColumns.length < 2) { // Minimum check for quantity and productName
            throw new IOException("Invalid CSV format: Header must contain at least 'quantity' and 'productName'.");
        }

        // Feature names start at index 2
        List<String> featureNames = new ArrayList<>(Arrays.asList(headerColumns).subList(2, headerColumns.length));

        if (featureNames.size() != expectedFeatureCount) {
            System.err.println("Warning: CSV header contains " + featureNames.size() +
                    " features, but expected " + expectedFeatureCount + ".");
        }


        // 2. Parse Data Rows
        return allLines.stream()
                .skip(1) // Skip header row
                .map(line -> {
                    // Use a simple split; this assumes no commas are present within quoted fields (like category names)
                    // If category names contain commas, a dedicated CSV parser library (like opencsv) is mandatory.
                    // The values are assumed to be quoted: ",value1,",",value2,"

                    String[] columns = line.split(",");

                    if (columns.length != headerColumns.length) {
                        // Skip malformed rows
                        System.err.println("Skipping malformed row with incorrect column count: " + line);
                        return null;
                    }

                    try {
                        TransactionDTO dto = new TransactionDTO();

                        // Column 0: quantity (int)
                        dto.setQuantity(Integer.parseInt(columns[0]));

                        // Column 1: productName (String)
                        dto.setProductName(columns[1]);

                        // Columns 2 onwards: Features (Map<String, String>)
                        Map<String, String> featureMap = new HashMap<>();
                        for (int i = 0; i < featureNames.size(); i++) {
                            String featureName = featureNames.get(i);
                            String rawValue = columns[i + 2];

                            // Remove possible leading/trailing quotes from the category name
                            String categoryValue = rawValue.replaceAll("^\"|\"$", "");

                            featureMap.put(featureName, categoryValue);
                        }
                        dto.setFeatureMap(featureMap);

                        return dto;

                    } catch (NumberFormatException e) {
                        System.err.println("Skipping row due to non-numeric quantity value: " + columns[0]);
                        return null;
                    } catch (Exception e) {
                        System.err.println("Skipping row due to unknown parsing error: " + line);
                        return null;
                    }
                })
                .filter(dto -> dto != null)
                .collect(Collectors.toList());
    }

    // --- Private Helper Methods ---

    /**
     * Converts a list of TransactionDTOs (categorized format) into a standard CSV string.
     */
    private String convertToCsv(List<TransactionDTO> transactions) {
        StringBuilder csv = new StringBuilder();
        TransactionDTO firstTransaction = transactions.get(0);

        // Header Row
        csv.append("quantity").append(CSV_SEPARATOR).append("productName");
        for (String featureName : firstTransaction.getFeatureMap().keySet()) {
            csv.append(CSV_SEPARATOR).append(featureName);
        }
        csv.append(LINE_SEPARATOR);

        // Data Rows
        for (TransactionDTO dto : transactions) {
            csv.append(dto.getQuantity()).append(CSV_SEPARATOR).append(dto.getProductName());

            for (String featureName : firstTransaction.getFeatureMap().keySet()) {
                String value = dto.getFeatureMap().getOrDefault(featureName, "");
                // Wrap value in quotes to handle categories that might contain commas
                csv.append(CSV_SEPARATOR).append("\"").append(value).append("\"");
            }
            csv.append(LINE_SEPARATOR);
        }
        return csv.toString();
    }

    /**
     * Converts a list of raw SyntheticTransaction objects (numeric/ID format) into a CSV string.
     */
    private String convertToRawCsv(List<SyntheticTransaction> transactions) {
        StringBuilder csv = new StringBuilder();

        // No header row is typically needed for raw PyTorch input data.

        for (SyntheticTransaction st : transactions) {
            // 1. Append continuous conditions (C1...C10)
            csv.append(st.getConditions().stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(CSV_SEPARATOR)));

            // 2. Append product ID (P1)
            csv.append(CSV_SEPARATOR).append(st.getProductId());

            // 3. Append quantity (Q1)
            csv.append(CSV_SEPARATOR).append(st.getQuantity());

            csv.append(LINE_SEPARATOR);
        }
        return csv.toString();
    }
}