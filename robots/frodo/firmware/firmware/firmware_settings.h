/*
 * firmware_setting.h
 *
 *  Created on: 3 Mar 2023
 *      Author: lehmann_workstation
 */

#ifndef FIRMWARE_SETTINGS_H_
#define FIRMWARE_SETTINGS_H_

#include "stm32h7xx.h"

#define FRODO_CONTROL_TASK_TIME_MS 50
#define FRODO_CONTROL_TASK_FREQUENCY 20
#define FRODO_SAMPLE_BUFFER_TIME 1

extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim3;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim5;
extern TIM_HandleTypeDef htim4;
extern TIM_HandleTypeDef htim15;
//extern DMA_HandleTypeDef hdma_tim4_ch1;
//extern DMA_HandleTypeDef hdma_tim4_ch2;

#define USE_FRODO_SHIELD

#ifdef USE_FRODO_SHIELD

#define MOTOR_LEFT_PWM_TIMER &htim15
#define MOTOR_LEFT_PWM_CHANNEL TIM_CHANNEL_1  // PE5
#define MOTOR_LEFT_DIR_PORT GPIOD
#define MOTOR_LEFT_DIR_PIN GPIO_PIN_5
#define MOTOR_LEFT_ENCODER_TIMER &htim2 // PA15
#define MOTOR_LEFT_DIRECTION -1
#define MOTOR_LEFT_INPUT_CAPTURE_TIMER &htim4
#define MOTOR_LEFT_INPUT_CAPTURE_TIMER_CHANNEL TIM_CHANNEL_1 // PB6

#define MOTOR_RIGHT_PWM_TIMER &htim15
#define MOTOR_RIGHT_PWM_CHANNEL TIM_CHANNEL_2  // PE6
#define MOTOR_RIGHT_DIR_PORT GPIOD
#define MOTOR_RIGHT_DIR_PIN GPIO_PIN_7
#define MOTOR_RIGHT_ENCODER_TIMER &htim3 // PD2
#define MOTOR_RIGHT_DIRECTION 1
#define MOTOR_RIGHT_INPUT_CAPTURE_TIMER &htim8
#define MOTOR_RIGHT_INPUT_CAPTURE_TIMER_CHANNEL TIM_CHANNEL_3 // PC8

#define MOTOR_INPUT_CAPTURE_TIMER_FREQUENCY 200e6
#define MOTOR_INPUT_CAPTURE_TIMER_PRESCALER 199
#define MOTOR_INPUT_CAPTURE_BUFFER_SIZE 3
#define FRODO_CONTROL_TASK_TICK_PORT GPIOD
#define FRODO_CONTROL_TASK_TICK_PIN GPIO_PIN_14

#endif
//#define FRODO_COMM_BUFFER_SIZE 10

#endif /* FIRMWARE_SETTINGS_H_ */
