///*
// * core_hardware_spi.cpp
// *
// *  Created on: 12 Mar 2023
// *      Author: Dustin Lehmann
// */
//
//#include "core_hardware_spi.h"
//
//core_hardware_SPI_slave *active_spi_slave = NULL;
//
///* ====================================================== */
//void spi_callback_rx(SPI_HandleTypeDef *hspi) {
//
//	if (active_spi_slave != NULL) {
//		active_spi_slave->rx_cmplt_function();
//	}
//}
//
///* ====================================================== */
//void spi_callback_tx(SPI_HandleTypeDef *hspi) {
//	if (active_spi_slave != NULL) {
//		active_spi_slave->tx_cmplt_function();
//	}
//}
//
///* ====================================================== */
//void spi_callback_rxtx(SPI_HandleTypeDef *hspi) {
//	if (active_spi_slave != NULL) {
//		active_spi_slave->rxtx_cmplt_function();
//	}
//}
//
///* ====================================================== */
//core_hardware_SPI_slave::core_hardware_SPI_slave() {
//
//}
//
///* ====================================================== */
//void core_hardware_SPI_slave::init(core_hardware_spi_config_t config) {
//	this->config = config;
//	active_spi_slave = this;
//
//	this->callbacks.rx_callback.registered = 0;
//	this->callbacks.tx_callback.registered = 0;
//	this->callbacks.rxtx_callback.registered = 0;
//
//	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_RX_COMPLETE_CB_ID,
//			spi_callback_rx);
//
//	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_TX_COMPLETE_CB_ID,
//			spi_callback_tx);
//
//	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_TX_RX_COMPLETE_CB_ID,
//			spi_callback_rxtx);
//}
///* ====================================================== */
//void core_hardware_SPI_slave::start() {
//
//}
//
//void core_hardware_SPI_slave::reset(){
//	HAL_SPI_Abort(this->config.hspi);
//	HAL_SPI_DMAStop(this->config.hspi);
//}
//
///* ====================================================== */
//void core_hardware_SPI_slave::receiveData(uint16_t len) {
//	HAL_SPI_Receive_DMA(this->config.hspi, this->config.rx_buffer, len);
//}
///* ====================================================== */
//void core_hardware_SPI_slave::receiveData(uint8_t *data, uint16_t len) {
//	HAL_SPI_Receive_DMA(this->config.hspi, data, len);
//}
///* ====================================================== */
//void core_hardware_SPI_slave::provideData(uint16_t len) {
//	HAL_SPI_Transmit_DMA(this->config.hspi, this->config.tx_buffer, len);
//}
///* ====================================================== */
//void core_hardware_SPI_slave::provideData(uint8_t *data, uint16_t len) {
//	HAL_SPI_Transmit_DMA(this->config.hspi, data, len);
//}
///* ====================================================== */
//void core_hardware_SPI_slave::receiveTransmitData(uint8_t *rx_buf,
//		uint8_t *tx_buf, uint16_t len) {
//	HAL_SPI_TransmitReceive_DMA(this->config.hspi, tx_buf, rx_buf, len);
//}
///* ====================================================== */
//void core_hardware_SPI_slave::registerCallback(
//		core_hardware_spi_callback_id_t callback_id,
//		core_utils_Callback<void, void> callback) {
//
//	switch (callback_id) {
//	case CORE_HARDWARE_SPI_CALLBACK_RX: {
//		this->callbacks.rx_callback = callback;
//		break;
//	}
//	case CORE_HARDWARE_SPI_CALLBACK_TX: {
//		this->callbacks.tx_callback = callback;
//		break;
//	}
//	case CORE_HARDWARE_SPI_CALLBACK_RXTX: {
//		this->callbacks.rxtx_callback = callback;
//		break;
//	}
//	}
//}
//
///* ====================================================== */
//void core_hardware_SPI_slave::rx_cmplt_function() {
//	// TODO
//	if (this->callbacks.rx_callback.registered) {
//		this->callbacks.rx_callback.call();
//	}
//}
///* ====================================================== */
//void core_hardware_SPI_slave::tx_cmplt_function() {
//	// TODO
//	if (this->callbacks.tx_callback.registered) {
//		this->callbacks.tx_callback.call();
//	}
//}
///* ====================================================== */
//void core_hardware_SPI_slave::rxtx_cmplt_function() {
//	// TODO
//	if (this->callbacks.rxtx_callback.registered) {
//		this->callbacks.rxtx_callback.call();
//	}
//}
///* ====================================================== */
///* ====================================================== */
//

/*
 * core_hardware_spi.cpp
 *
 *  Created on: 12 Mar 2023
 *      Author: Dustin Lehmann
 */

#include "core_hardware_spi.h"
#include "stm32h7xx_hal.h"

static core_hardware_SPI_slave *active_spi_slave = nullptr;

static void spi_callback_rx(SPI_HandleTypeDef *hspi);
static void spi_callback_tx(SPI_HandleTypeDef *hspi);
static void spi_callback_rxtx(SPI_HandleTypeDef *hspi);
static void spi_callback_error(SPI_HandleTypeDef *hspi);

/* ====================================================== */
static void spi_callback_rx(SPI_HandleTypeDef *hspi) {
	if (active_spi_slave)
		active_spi_slave->rx_cmplt_function();
}

static void spi_callback_tx(SPI_HandleTypeDef *hspi) {
	if (active_spi_slave)
		active_spi_slave->tx_cmplt_function();
}

static void spi_callback_rxtx(SPI_HandleTypeDef *hspi) {
	if (active_spi_slave) {
		active_spi_slave->rxtx_cmplt_function();
		active_spi_slave->tx_cmplt_function();
	}
}

static void spi_callback_error(SPI_HandleTypeDef *hspi) {
	if (active_spi_slave)
		active_spi_slave->error_function();
}

/* ====================================================== */
core_hardware_SPI_slave::core_hardware_SPI_slave() {
}

/* ====================================================== */
void core_hardware_SPI_slave::init(core_hardware_spi_config_t config) {
	this->config = config;
	active_spi_slave = this;

	this->callbacks.rx_callback.registered = 0;
	this->callbacks.tx_callback.registered = 0;
	this->callbacks.rxtx_callback.registered = 0;
	this->callbacks.error_callback.registered = 0;

	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_RX_COMPLETE_CB_ID,
			spi_callback_rx);
	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_TX_COMPLETE_CB_ID,
			spi_callback_tx);
	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_TX_RX_COMPLETE_CB_ID,
			spi_callback_rxtx);
	HAL_SPI_RegisterCallback(this->config.hspi, HAL_SPI_ERROR_CB_ID,
			spi_callback_error);
}

void core_hardware_SPI_slave::start() {
	// User decides when to arm first RX
}

void core_hardware_SPI_slave::reset() {
	HAL_SPI_Abort(this->config.hspi);
	HAL_SPI_DMAStop(this->config.hspi);
//
//	__HAL_SPI_DISABLE(this->config.hspi);
//	drain_rx_fifo_();
//	__HAL_SPI_CLEAR_OVRFLAG(this->config.hspi);
//	__HAL_SPI_ENABLE(this->config.hspi);
}

/* ====================================================== */
void core_hardware_SPI_slave::receiveData(uint16_t len) {
	HAL_SPI_Receive_DMA(this->config.hspi, this->config.rx_buffer, len);
}

void core_hardware_SPI_slave::receiveData(uint8_t *data, uint16_t len) {
	HAL_SPI_Receive_DMA(this->config.hspi, data, len);
}

/* ====================================================== */
// Caller must provide dummy_rx buffer of the same length as tx_buf
void core_hardware_SPI_slave::provideData(uint8_t *tx_buf, uint16_t len,
		uint8_t *dummy_rx) {
	HAL_SPI_TransmitReceive_DMA(this->config.hspi, tx_buf, dummy_rx, len);
}

/* ====================================================== */
void core_hardware_SPI_slave::receiveTransmitData(uint8_t *rx_buf,
		uint8_t *tx_buf, uint16_t len) {
	HAL_SPI_TransmitReceive_DMA(this->config.hspi, tx_buf, rx_buf, len);
}

/* ====================================================== */
void core_hardware_SPI_slave::registerCallback(
		core_hardware_spi_callback_id_t callback_id,
		core_utils_Callback<void, void> callback) {
	switch (callback_id) {
	case CORE_HARDWARE_SPI_CALLBACK_RX:
		this->callbacks.rx_callback = callback;
		break;
	case CORE_HARDWARE_SPI_CALLBACK_TX:
		this->callbacks.tx_callback = callback;
		break;
	case CORE_HARDWARE_SPI_CALLBACK_RXTX:
		this->callbacks.rxtx_callback = callback;
		break;
	case CORE_HARDWARE_SPI_CALLBACK_ERROR:
		this->callbacks.error_callback = callback;
		break;
	default:
		break;
	}
}

/* ====================================================== */
void core_hardware_SPI_slave::rx_cmplt_function() {
	if (this->callbacks.rx_callback.registered)
		this->callbacks.rx_callback.call();
}

void core_hardware_SPI_slave::tx_cmplt_function() {
	if (this->callbacks.tx_callback.registered)
		this->callbacks.tx_callback.call();
}

void core_hardware_SPI_slave::rxtx_cmplt_function() {
	if (this->callbacks.rxtx_callback.registered)
		this->callbacks.rxtx_callback.call();
}

void core_hardware_SPI_slave::error_function() {
	__HAL_SPI_DISABLE(this->config.hspi);
	drain_rx_fifo_();
	__HAL_SPI_CLEAR_OVRFLAG(this->config.hspi);
	__HAL_SPI_ENABLE(this->config.hspi);

	HAL_SPI_Abort(this->config.hspi);

	if (this->callbacks.error_callback.registered)
		this->callbacks.error_callback.call();
}

/* ====================================================== */
void core_hardware_SPI_slave::drain_rx_fifo_() {
	while (__HAL_SPI_GET_FLAG(this->config.hspi, SPI_FLAG_RXP)) {
		(void) this->config.hspi->Instance->RXDR;
	}
}
