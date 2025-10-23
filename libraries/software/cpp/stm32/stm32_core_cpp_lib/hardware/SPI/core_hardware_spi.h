///*
// * core_hardware_spi.h
// *
// *  Created on: 12 Mar 2023
// *      Author: Dustin Lehmann
// */
//
//#ifndef HARDWARE_SPI_CORE_HARDWARE_SPI_H_
//#define HARDWARE_SPI_CORE_HARDWARE_SPI_H_
//
//#include "core.h"
//
//typedef enum core_hardware_spi_callback_id_t {
//	CORE_HARDWARE_SPI_CALLBACK_RX,
//	CORE_HARDWARE_SPI_CALLBACK_TX,
//	CORE_HARDWARE_SPI_CALLBACK_RXTX,
//} core_hardware_spi_callback_id_t;
//
//typedef struct core_hardware_spi_config_t {
//	SPI_HandleTypeDef *hspi;
//	uint8_t *rx_buffer;
//	uint8_t *tx_buffer;
//} core_hardware_spi_config_t;
//
//typedef struct core_hardware_spi_callbacks_t {
//	core_utils_Callback<void, void> rx_callback;
//	core_utils_Callback<void, void> tx_callback;
//	core_utils_Callback<void, void> rxtx_callback;
//} core_hardware_spi_callbacks_t;
//
//class core_hardware_SPI_slave {
//public:
//	core_hardware_SPI_slave();
//
//	void init(core_hardware_spi_config_t config);
//	void start();
//
//	void reset();
//
//	void registerCallback(core_hardware_spi_callback_id_t callback_id,
//			core_utils_Callback<void, void> callback);
//
//	void receiveData(uint16_t len);
//	void receiveData(uint8_t *buffer, uint16_t len);
//
//	void provideData(uint16_t len);
//	void provideData(uint8_t *buffer, uint16_t len);
//
//	void receiveTransmitData(uint8_t *rx_buf, uint8_t *tx_buf, uint16_t len);
//
//	void rx_cmplt_function();
//	void tx_cmplt_function();
//	void rxtx_cmplt_function();
//
//	core_hardware_spi_config_t config;
//	core_hardware_spi_callbacks_t callbacks;
//
//private:
//
//};
//
//#endif /* HARDWARE_SPI_CORE_HARDWARE_SPI_H_ */

/*
 * core_hardware_spi.h
 *
 *  Created on: 12 Mar 2023
 *      Author: Dustin Lehmann
 */

#ifndef HARDWARE_SPI_CORE_HARDWARE_SPI_H_
#define HARDWARE_SPI_CORE_HARDWARE_SPI_H_

#include "core.h"

typedef enum core_hardware_spi_callback_id_t {
	CORE_HARDWARE_SPI_CALLBACK_RX,
	CORE_HARDWARE_SPI_CALLBACK_TX,
	CORE_HARDWARE_SPI_CALLBACK_RXTX,
	CORE_HARDWARE_SPI_CALLBACK_ERROR,
} core_hardware_spi_callback_id_t;

typedef struct core_hardware_spi_config_t {
	SPI_HandleTypeDef *hspi;
	uint8_t *rx_buffer;
	uint8_t *tx_buffer;
} core_hardware_spi_config_t;

typedef struct core_hardware_spi_callbacks_t {
	core_utils_Callback<void, void> rx_callback;
	core_utils_Callback<void, void> tx_callback;
	core_utils_Callback<void, void> rxtx_callback;
	core_utils_Callback<void, void> error_callback;
} core_hardware_spi_callbacks_t;

class core_hardware_SPI_slave {
public:
	core_hardware_SPI_slave();

	void init(core_hardware_spi_config_t config);
	void start();
	void reset();

	void registerCallback(core_hardware_spi_callback_id_t callback_id,
			core_utils_Callback<void, void> callback);

	// RX-only
	void receiveData(uint16_t len);
	void receiveData(uint8_t *buffer, uint16_t len);

	// TXRX (caller provides dummy RX buffer of same length)
	void provideData(uint8_t *tx_buf, uint16_t len, uint8_t *dummy_rx);

	// Generic full-duplex
	void receiveTransmitData(uint8_t *rx_buf, uint8_t *tx_buf, uint16_t len);

	// Callback fan-out
	void rx_cmplt_function();
	void tx_cmplt_function();
	void rxtx_cmplt_function();
	void error_function();

	core_hardware_spi_config_t config;
	core_hardware_spi_callbacks_t callbacks;

private:
	void drain_rx_fifo_();
};

#endif /* HARDWARE_SPI_CORE_HARDWARE_SPI_H_ */
