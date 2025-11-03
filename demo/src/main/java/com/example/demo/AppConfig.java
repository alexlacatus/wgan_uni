package com.example.demo;// src/main/java/com/example/demo/config/AppConfig.java (sau în directorul principal)


import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AppConfig {

    /**
     * Definește RestTemplate ca un Spring Bean.
     * Acesta permite injectarea automată (Autowiring) în alte servicii (GenerationService).
     */
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}