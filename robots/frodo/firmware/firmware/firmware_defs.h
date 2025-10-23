/*
 * firmware_defs.h
 *
 *  Created on: 16 Mar 2023
 *      Author: lehmann_workstation
 */

#ifndef FIRMWARE_DEFS_H_
#define FIRMWARE_DEFS_H_

#include "core.h"
#include "firmware_settings.h"


#define TWIPR_REGISTER_MAP_GENERAL 0x01

// Register Maps and Messages on the CM4
#define CM4_STANDARD_REGISTER_MAP 0x01

// Messages
#define TWIPR_MESSAGE_EVENT_WARNING 0x0301


// Register Addresses
#define REG_ADDRESS_R_FIRMWARE_STATE 0x01
#define REG_ADDRESS_R_FIRMWARE_TICK 0x02
#define REG_ADDRESS_F_SET_SPEED 0x03

//#define REG_ADDRESS_R_FIRMWARE_REVISION 0x03
//#define REG_ADDRESS_F_FIRMWARE_DEBUGFUNCTION 0x04
#define REG_ADDRESS_F_FIRMWARE_BEEP 0x05
//#define REG_ADDRESS_R_BOARD_REVISION 0x06
//
#define REG_ADDRESS_F_EXTERNAL_LED 0x07
#define REG_ADDRESS_F_ALL_EXTERNAL_LEDS 0x09




#define FRODO_MESSAGE_ID_SAMPLE_STREAM 0x10


// ------------------------------------------------------------------------------------------------ //
typedef struct twipr_firmware_revision_t {
	uint8_t major;
	uint8_t minor;
}twipr_firmware_revision_t;

// ------------------------------------------------------------------------------------------------ //
typedef enum twipr_error_t {
	TWIPR_ERROR_NONE = 0,
	TWIPR_ERROR_CRITICAL = 1,
	TWIPR_ERROR_WARNING = 2,
} twipr_error_t;

typedef enum twipr_supervisor_error_t {
	TWIPR_SUPERVISOR_NONE = 0,
} twipr_supervisor_error_t;

typedef enum twipr_firmware_state_t {
	TWIPR_FIRMWARE_STATE_ERROR = -1,
	TWIPR_FIRMWARE_STATE_IDLE = 0,
	TWIPR_FIRMWARE_STATE_RUNNING = 1,
	TWIPR_FIRMWARE_STATE_RESET = 2,
} twipr_firmware_state_t;

#define FRODO_FIRMWARE_SAMPLE_BUFFER_SIZE (uint16_t) (FRODO_SAMPLE_BUFFER_TIME * 1000 / FRODO_CONTROL_TASK_TIME_MS)


extern DMA_HandleTypeDef hdma_memtomem_dma2_stream0;
#define TWIPR_FIRMWARE_SAMPLE_DMA_STREAM &hdma_memtomem_dma2_stream0

#endif /* FIRMWARE_DEFS_H_ */

