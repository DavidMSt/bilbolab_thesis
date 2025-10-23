/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "firmware_c.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define LED2_Pin GPIO_PIN_3
#define LED2_GPIO_Port GPIOE
#define MOTOR_RIGHT_DIR_Pin GPIO_PIN_0
#define MOTOR_RIGHT_DIR_GPIO_Port GPIOC
#define MOTOR_LEFT_DIR_Pin GPIO_PIN_1
#define MOTOR_LEFT_DIR_GPIO_Port GPIOC
#define MOTOR_RIGHT_PWM_Pin GPIO_PIN_0
#define MOTOR_RIGHT_PWM_GPIO_Port GPIOA
#define MOTOR_LEFT_PWM_Pin GPIO_PIN_1
#define MOTOR_LEFT_PWM_GPIO_Port GPIOA
#define UART_RESET_INTERRUPT_Pin GPIO_PIN_2
#define UART_RESET_INTERRUPT_GPIO_Port GPIOB
#define UART_RESET_INTERRUPT_EXTI_IRQn EXTI2_IRQn
#define NEW_SAMPLES_OUTPUT_Pin GPIO_PIN_9
#define NEW_SAMPLES_OUTPUT_GPIO_Port GPIOE
#define LED1_Pin GPIO_PIN_15
#define LED1_GPIO_Port GPIOE
#define CS_IMU_Pin GPIO_PIN_12
#define CS_IMU_GPIO_Port GPIOB
#define CONTROL_TASK_TICK_Pin GPIO_PIN_14
#define CONTROL_TASK_TICK_GPIO_Port GPIOD
#define RS485_EN_Pin GPIO_PIN_15
#define RS485_EN_GPIO_Port GPIOD
#define ENCODER_RIGHT_INPUT_CAPTURE_Pin GPIO_PIN_8
#define ENCODER_RIGHT_INPUT_CAPTURE_GPIO_Port GPIOC
#define ENCODER_LEFT_INPUT_CAPTURE_Pin GPIO_PIN_6
#define ENCODER_LEFT_INPUT_CAPTURE_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
