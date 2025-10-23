/*
 * twipr_logging.h
 *
 *  Created on: Mar 7, 2023
 *      Author: lehmann_workstation
 */

#ifndef LOGGING_FRODO_LOGGING_H_
#define LOGGING_FRODO_LOGGING_H_

#include "stm32h7xx.h"
#include "frodo_drive.h"
#include "firmware_defs.h"

//typedef struct frodo_sample_t {
//	uint32_t tick;
//	uint8_t debug;
//	uint8_t state;
//	float battery_voltage;
//
//	float goal_speed_l;
//	float goal_speed_r;
//	float rpm_l;
//	float rpm_r;
//	float velocity_l;
//	float velocity_r;
//	float velocity_forward;
//	float velocity_turn;
//
//	float imu_gyr_x;
//	float imu_gyr_y;
//	float imu_gyr_z;
//	float imu_acc_x;
//	float imu_acc_y;
//	float imu_acc_z;
//
//} frodo_sample_t;

class FRODO_Firmware;

typedef struct frodo_general_sample_t {
	uint32_t tick;
	uint8_t state;
	float update_time;
	float battery_voltage;
} frodo_general_sample_t;

typedef struct frodo_sample_t {
	frodo_general_sample_t general;
	frodo_drive_sample_t drive;
} frodo_sample_t;

typedef struct frodo_logging_config_t {
	FRODO_Firmware *firmware;
	FRODO_Drive *drive;
	bool use_buffer;
} frodo_logging_config_t;

typedef enum frodo_logging_buffer_status_t {
	FRODO_LOGGING_BUFFER_FULL = 1, FRODO_LOGGING_BUFFER_NOT_FULL = 0,
} frodo_logging_buffer_status_t;


class FRODO_Logging {
public:
	FRODO_Logging();
	void init(frodo_logging_config_t);
	void start();

	frodo_logging_buffer_status_t collectSamples();

	frodo_sample_t getCurrentSample();

	frodo_sample_t sample_buffer[FRODO_FIRMWARE_SAMPLE_BUFFER_SIZE];

	frodo_sample_t current_sample;
	frodo_logging_config_t config;

private:

	uint32_t sample_index = 0;

};

#endif /* LOGGING_FRODO_LOGGING_H_ */
