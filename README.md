# IoT-Enabled Drip Irrigation System

## Overview
This project is an IoT-enabled drip irrigation system designed to optimize water usage through automated and manual control modes. The system integrates multiple moisture sensors, a motorized pump, and a web interface for remote monitoring and control. Data is stored in a database for analysis and decision-making.

## Features
- **Manual Mode**: Users can manually turn the pump ON/OFF via buttons.
- **Automatic Mode**: The pump is controlled based on moisture sensor data:
  - If even one sensor detects moisture, the pump remains OFF.
  - If all sensors detect dryness, the pump automatically turns ON.
- **Web-Based Control**: Users can override automatic mode and control the pump via a website.
- **Real-Time Monitoring**: Temperature, humidity, and sensor statuses are displayed on an LCD screen.
- **Data Storage**: Sensor readings and pump status are logged in a database for analysis.

## Hardware Implementation
### Components Used:
- ESP32 Microcontroller
- Moisture Sensors
- DHT11 Temperature and Humidity Sensor
- 16x2 LCD Display
- Relay Module for Motor Control
- WiFi Module (Built-in ESP32)

### Circuit Connections:
- Moisture Sensors connected to GPIO pins
- DHT11 Sensor connected to pin 23
- LCD Display connected to pins 13, 12, 14, 27, 26, 25
- Motor Pump controlled via pin 5
- Mode Switch connected to pin 17

### Code Implementation
The system is programmed in C++ using the Arduino framework.
```cpp
#include <LiquidCrystal.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "DHTesp.h"
#define DHTpin 23

LiquidCrystal lcd(13, 12, 14, 27, 26, 25);
HTTPClient http;
const char *ssid = "iotserver";
const char *password = "iotserver123";
String servername = "http://projectsfactoryserver.in/storedata.php?name=";
String accountname = "iot960";
...
```
(Full code available in the repository)

## Website Interface
The system is integrated with a web server for remote monitoring and control. The website URL is:
[IoT Drip Irrigation System](http://projectsfactoryserver.in/)

### Web Features:
- Displays real-time sensor data
- Allows manual pump control
- Logs historical data for analysis

## How It Works
1. **Power ON** the system.
2. **Connect** the ESP32 to WiFi.
3. The system reads **temperature, humidity, and moisture sensor data**.
4. In **Automatic Mode**, the pump operates based on sensor inputs.
5. In **Manual Mode**, users can control the pump via buttons or the website.
6. Data is **stored and updated** on the web server for real-time access.

## Future Enhancements
- Integration with a mobile app for easier access.
- Advanced data analytics for better irrigation predictions.
- Solar-powered operation for energy efficiency.

## Contributors
Developed by S V V Ramesh
             T Bhavitha Sri

