package com.example.demo.helper;

import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.List;

import static com.example.demo.helper.TransactionMapper.*;

@Component
public class FeatureMappers {
    public static String getForLabels(float value, List<String> categories) {
        if (categories == null || categories.isEmpty()) {
            return "NO_BINS_DEFINED";
        }

        int numBins = categories.size();

        float clampedValue = Math.max(FEATURE_RANGE_MIN, Math.min(FEATURE_RANGE_MAX, value));
        float normalizedValue = (clampedValue - FEATURE_RANGE_MIN) / FEATURE_RANGE_SIZE;

        // Use epsilon to ensure max value lands in the last bin
        int binIndex = (int) Math.floor((normalizedValue - 1e-6f) * numBins);

        binIndex = Math.max(0, Math.min(numBins - 1, binIndex));

        return categories.get(binIndex);
    }

    public static String getForDate(float value, List<String> categories) {
        // --- Input Validation and Setup ---
        if (categories == null || categories.size() != 2) {
            return "DATE_RANGE_ERROR: Categories must contain exactly two date strings [start, end]";
        }

        Instant startDate;
        Instant endDate;
        try {
            startDate = Instant.parse(categories.get(0));
            endDate = Instant.parse(categories.get(1));
        } catch (DateTimeParseException e) {
            return "DATE_PARSE_ERROR: Dates must be in a valid format (e.g., ISO 8601)";
        }

        if (startDate.isAfter(endDate)) {
            return "DATE_RANGE_ERROR: Start date cannot be after end date";
        }

        float clampedValue = Math.max(FEATURE_RANGE_MIN, Math.min(FEATURE_RANGE_MAX, value));

        float normalizedValue = (clampedValue - FEATURE_RANGE_MIN) / FEATURE_RANGE_SIZE;

        long startMillis = startDate.toEpochMilli();
        long endMillis = endDate.toEpochMilli();
        long durationMillis = endMillis - startMillis;

        // Linearly interpolates the date: start + (duration * normalized_value)
        long resultMillis = (long) (startMillis + (durationMillis * normalizedValue));
        Instant resultDate = Instant.ofEpochMilli(resultMillis);

        return resultDate.toString();
    }

    public static String getForPhoneNumber(float value, List<String> categories) {

        float clampedValue = Math.max(FEATURE_RANGE_MIN, Math.min(FEATURE_RANGE_MAX, value));
        float normalizedValue = (clampedValue - FEATURE_RANGE_MIN) / FEATURE_RANGE_SIZE;

        final long MAX_9_DIGIT_VALUE_PLUS_1 = 1_000_000_000L;

        // Linearly interpolate the 9-digit number
        long mappedNumber = (long) (normalizedValue * MAX_9_DIGIT_VALUE_PLUS_1);



        // Format the 9-digit number with leading zeros if necessary
        String nineDigits = String.format("%09d", mappedNumber % MAX_9_DIGIT_VALUE_PLUS_1);

        return "0" + nineDigits;
    }
    public static String getForNumber(float value, List<String> categories) {
        if (categories == null || categories.size() < 3) {
            return "NUMBER_RANGE_ERROR: Categories must contain at least three values: [lower, upper, precision]";
        }

        double lowerBound;
        double upperBound;
        int precision;

        try {
            lowerBound = Double.parseDouble(categories.get(0));
            upperBound = Double.parseDouble(categories.get(1));
            precision = Integer.parseInt(categories.get(2));
        } catch (NumberFormatException e) {
            return "NUMBER_PARSE_ERROR: Bounds and precision must be valid numbers.";
        }


        if (upperBound < lowerBound) {
            return "NUMBER_RANGE_ERROR: Upper bound cannot be less than lower bound.";
        }

        if (precision < 0) {
            return "NUMBER_PARSE_ERROR: Precision must be a non-negative integer.";
        }

        float clampedValue = Math.max(FEATURE_RANGE_MIN, Math.min(FEATURE_RANGE_MAX, value));

        // Normalizes the clamped value to the range [0.0, 1.0]
        float normalizedValue = (clampedValue - FEATURE_RANGE_MIN) / FEATURE_RANGE_SIZE;

        // Calculate the range size
        double rangeSize = upperBound - lowerBound;

        double resultNumber = lowerBound + (rangeSize * normalizedValue);
        String formatString = "%." + precision + "f";

        // Format the resulting number
        return String.format(formatString, resultNumber);
    }
    public static String getForCardNumber(float value, List<String> categories) {

        float clampedValue = Math.max(FEATURE_RANGE_MIN, Math.min(FEATURE_RANGE_MAX, value));

        float normalizedValue = (clampedValue - FEATURE_RANGE_MIN) / FEATURE_RANGE_SIZE;
        final long MAX_10_DIGIT_VALUE_PLUS_1 = 10_000_000_000L;

        long mappedNumber = (long) (normalizedValue * MAX_10_DIGIT_VALUE_PLUS_1);

        // Use the modulo operator to ensure the number remains within the 10-digit capacity
        long numberToFormat = mappedNumber % MAX_10_DIGIT_VALUE_PLUS_1;

        // Format the number to exactly 10 digits, padding with leading zeros if necessary
        return String.format("%010d", numberToFormat);
    }
}
