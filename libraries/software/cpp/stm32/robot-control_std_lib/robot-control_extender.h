/*
 * robot-control_extender.h
 *
 *  Created on: Apr 24, 2024
 *      Author: Dustin Lehmann
 */

#ifndef ROBOT_CONTROL_EXTENDER_H_
#define ROBOT_CONTROL_EXTENDER_H_

#include <core.h>
#include <robot-control_extender_registers.h>

#define EXTENDER_ADDRESS 0x02

typedef struct extender_config_struct_t {
	I2C_HandleTypeDef *hi2c;
} extender_config_struct_t;

typedef struct rgb_color_struct_t {
	uint8_t red;
	uint8_t green;
	uint8_t blue;
} rgb_color_struct_t;

typedef struct external_led_colors_struct_t {
	rgb_color_struct_t colors[16];
} external_led_colors_struct_t;

class RobotControl_Extender {
public:
	RobotControl_Extender();

	void init(extender_config_struct_t config);
	void start();

	void setStatusLED(int8_t status);

	// Internal RGB (3 discrete LEDs on the extender)
	void rgbLED_intern_setMode(uint8_t position, uint8_t mode); // 0=continuous, 1=blink
	void rgbLED_intern_setColor(uint8_t position, uint8_t red, uint8_t green,
			uint8_t blue);
	void rgbLED_intern_setState(uint8_t position, uint8_t state); // 0=off, 1=on (MSB)
	void rgbLED_intern_blink(uint8_t position, uint16_t on_time_ms);

	// External 16-LED WS2812 strip — PER-LED control (color only)
	// Fill: set all 16 LEDs to the same color
	void rgbLEDStrip_extern_setColor(rgb_color_struct_t color);
	// Set one pixel [0..15]
	void rgbLEDStrip_extern_setPixelColor(uint8_t index, uint8_t red,
			uint8_t green, uint8_t blue);
	void rgbLEDStrip_extern_setPixelColor(uint8_t index,
			rgb_color_struct_t color);
	// Bulk: apply colors[0..count-1] to LEDs 0..count-1 (count <= 16)
	void rgbLEDStrip_extern_setAllColors(external_led_colors_struct_t colors);

	// Buzzer
	void buzzer_setConfig(float frequency, uint16_t on_time, uint8_t repeats);
	void buzzer_start();

	bool readBatteryVoltage();

	float getBatteryVoltage() const { return battery_voltage; }

	extender_config_struct_t config;
	external_led_colors_struct_t current_external_colors;


	float battery_voltage = 0.0f;

private:

};

#endif /* ROBOT_CONTROL_EXTENDER_H_ */
