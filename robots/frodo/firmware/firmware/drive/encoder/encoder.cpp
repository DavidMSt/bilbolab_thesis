/*
 * encoder.cpp
 *
 *  Created on: May 7, 2024
 *      Author: Dustin Lehmann
 */

#include "encoder.h"

Encoder::Encoder() {

}

/* -------------------------------------------------------------------- */
void Encoder::init(encoder_config_t config) {
	this->config = config;
	this->reading_index = 0;

}
/* -------------------------------------------------------------------- */
void Encoder::start() {
	HAL_TIM_Base_Start(this->config.tim);
}

/* -------------------------------------------------------------------- */
void Encoder::update() {
	int value = __HAL_TIM_GET_COUNTER(this->config.tim);
	if (value > 0) {
		nop();
	}
	this->readings[reading_index] = value;
	__HAL_TIM_SET_COUNTER(this->config.tim, 0);
	this->_updateIndex();
}

/* -------------------------------------------------------------------- */
/* return Rotations per Minute */
float Encoder::getRPM() {
	int sum = 0;
	for (int i = 0; i < WINDOW_SIZE; i++) {
		sum += this->readings[i];
	}
	float average_steps_per_update = (float) sum / (float) WINDOW_SIZE;

	float rounds_per_second = (average_steps_per_update
			* (1000 / this->config.update_time_ms))
			/ (IMPULSE_PER_ROTATION * REDUCTION_RATIO);

	return rounds_per_second * 60;
}

/* -------------------------------------------------------------------- */
/* return Velocity in mm/s */
float Encoder::getVelocity() {
	return (getRPM() / 60) * 2 * M_PI * WHEEL_RADIUS_MM
			* this->config.velocity_scale;
}

/* -------------------------------------------------------------------- */
void Encoder::_updateIndex() {
	this->reading_index++;
	if (reading_index >= WINDOW_SIZE) {
		reading_index = 0;
	}

}

InputCaptureEncoder *input_capture_encoders[MAX_ENCODERS] = { };
int index_encoders = 0;

/* ======================================================================= */
InputCaptureEncoder::InputCaptureEncoder() {

}

/* --------------------------------------------------------------------- */
void InputCaptureEncoder::init(input_capture_encoder_config_t config) {
	this->config = config;

	input_capture_encoders[index_encoders] = this;
	index_encoders++;

	HAL_TIM_RegisterCallback(this->config.htim, HAL_TIM_PERIOD_ELAPSED_CB_ID,
			tim_period_elapsed_callback);
	HAL_TIM_RegisterCallback(this->config.htim, HAL_TIM_IC_CAPTURE_CB_ID,
			tim_ic_callback);

	float effective_frequency = (float) (this->config.timer_frequency)
			/ (float) (this->config.timer_prescaler + 1);

	this->_tick_time = 1.0 / effective_frequency;
}

/* --------------------------------------------------------------------- */
void InputCaptureEncoder::start() {
	HAL_TIM_IC_Start_IT(this->config.htim, this->config.channel);
}

/* --------------------------------------------------------------------- */
void InputCaptureEncoder::ic_interrupt_handler(uint16_t value) {

	// Reset the timer
	__HAL_TIM_SetCounter(this->config.htim, 0);

	this->_appendValue((uint32_t) value);

}

/* --------------------------------------------------------------------- */
void InputCaptureEncoder::timer_overflow_handler() {
	// Reset the timer
	__HAL_TIM_SetCounter(this->config.htim, 0);

	this->_appendValue((uint32_t) ((float) 0xFFFFFFFF / (float) MOTOR_INPUT_CAPTURE_BUFFER_SIZE));
}

/* --------------------------------------------------------------------- */
void InputCaptureEncoder::_appendValue(uint32_t value) {
	this->value_buffer[this->buffer_index] = value;

	this->buffer_index++;

	if (this->buffer_index == MOTOR_INPUT_CAPTURE_BUFFER_SIZE) {
		this->buffer_index = 0;
	}
}

/* --------------------------------------------------------------------- */
uint32_t InputCaptureEncoder::_getMeanValue() {

	uint32_t sum = 0;

	__disable_irq();

	for (int i=0; i<MOTOR_INPUT_CAPTURE_BUFFER_SIZE; i++){
		sum += this->value_buffer[i];
	}
	__enable_irq();

	return sum / MOTOR_INPUT_CAPTURE_BUFFER_SIZE;
}

/* --------------------------------------------------------------------- */
float InputCaptureEncoder::getRPM() {

	uint32_t timer_value = this->_getMeanValue();

	// Ensure timer_value, tick_time, and ticks_per_revolution are valid
	if (timer_value == 0 || this->_tick_time <= 0
			|| this->config.ticks_per_revolution <= 0) {
		return 0.0;  // Return 0 RPM if inputs are invalid
	}

	// Calculate the time in seconds for the input capture

	float event_frequency = 1.0 / (timer_value * this->_tick_time);

	float rounds_per_seconds = event_frequency
			/ this->config.ticks_per_revolution;

	return rounds_per_seconds * 60.0;
}

/* --------------------------------------------------------------------- */
float InputCaptureEncoder::getVelocity(){
	return (this->getRPM() / 60) * 2 * M_PI * WHEEL_RADIUS_MM;
}

/* =========================================================================== */

void tim_period_elapsed_callback(TIM_HandleTypeDef *htim) {
	for (int i = 0; i < index_encoders; i++) {
		if (input_capture_encoders[i]->config.htim == htim) {
			input_capture_encoders[i]->timer_overflow_handler();
		}
	}
}

/* =========================================================================== */
void tim_ic_callback(TIM_HandleTypeDef *htim) {
for (int i = 0; i < index_encoders; i++) {
	if (input_capture_encoders[i]->config.htim == htim) {
		// Get the configured channel for this encoder
		uint32_t configured_channel = input_capture_encoders[i]->config.channel;

		// Check if the interrupt corresponds to the configured channel
		if ((configured_channel == TIM_CHANNEL_1)
				&& __HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC1) &&
				__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC1)) {

			// Clear the interrupt flag
			__HAL_TIM_CLEAR_IT(htim, TIM_IT_CC1);

			// Call the handler with the captured value
			uint16_t captured_value = __HAL_TIM_GET_COMPARE(htim,
					TIM_CHANNEL_1);
			input_capture_encoders[i]->ic_interrupt_handler(captured_value);

		} else if ((configured_channel == TIM_CHANNEL_2)
				&& __HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC2) &&
				__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC2)) {

			// Clear the interrupt flag
			__HAL_TIM_CLEAR_IT(htim, TIM_IT_CC2);

			// Call the handler with the captured value
			uint16_t captured_value = __HAL_TIM_GET_COMPARE(htim,
					TIM_CHANNEL_2);
			input_capture_encoders[i]->ic_interrupt_handler(captured_value);

		} else if ((configured_channel == TIM_CHANNEL_3)
				&& __HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC3) &&
				__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC3)) {

			// Clear the interrupt flag
			__HAL_TIM_CLEAR_IT(htim, TIM_IT_CC3);

			// Call the handler with the captured value
			uint16_t captured_value = __HAL_TIM_GET_COMPARE(htim,
					TIM_CHANNEL_3);
			input_capture_encoders[i]->ic_interrupt_handler(captured_value);

		} else if ((configured_channel == TIM_CHANNEL_4)
				&& __HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC4) &&
				__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC4)) {

			// Clear the interrupt flag
			__HAL_TIM_CLEAR_IT(htim, TIM_IT_CC4);

			// Call the handler with the captured value
			uint16_t captured_value = __HAL_TIM_GET_COMPARE(htim,
					TIM_CHANNEL_4);
			input_capture_encoders[i]->ic_interrupt_handler(captured_value);
		}

		// Stop searching once we find the matching encoder
		break;
	}
}
}

