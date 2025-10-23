<script lang="ts">
	import { shortcut } from '$lib/helpers/shortcut';
	import { get } from 'svelte/store';
	import { messages, getCurrentTimestamp, sendMessage } from '$lib/stores/messages';

	let armed = false;
	let stop = false;

	function pressed(a: boolean) {
		const msgs = get(messages);
		if (a) {
			console.log('Emergency stop pressed');
			stop = true;

			const message = {
                "type" : "command",
                data : {
                    command : "emergency"
                }
			};
			sendMessage(message);

			setTimeout(() => {
				stop = false;
				console.log('Emergency stop released');
			}, 1000);
		} else {
			armed = true;
			console.log('Emergency stop armed');
			setTimeout(() => {
				armed = false;
				console.log('Emergency stop disarmed');
			}, 1000);
		}
	}
</script>

<button
	class="absolute bottom-0 right-0 h-[32vh] w-[30vh] translate-x-[49%] translate-y-[53%] -rotate-[40deg] scale-50 rounded-3xl
border-none
bg-amber-500 bg-opacity-0 text-center
text-white
outline-0 outline-white transition-all duration-100
ease-in-out hover:translate-x-[2%]
hover:translate-y-[10%] hover:-rotate-12
 hover:scale-150
 hover:bg-opacity-100
 hover:text-amber-900
 hover:opacity-100
hover:outline
 hover:outline-8

 "
	class:armed
	on:click={() => pressed(true)}
	use:shortcut={{ code: 'Space', callback: () => pressed(armed), allowDefault: true }}
>
	<span class="font-bold">EMERGENCY <br /><span class="text-5xl">STOP</span></span>
	<div
		class="m-auto h-[22vh] w-[22vh] rounded-full border-2 border-red-700 bg-red-500 shadow-2xl"
		class:bg-red-700={stop}
		class:border-red-900={stop}
		class:scale-[98%]={stop}
	></div>
</button>

<style>
	.armed {
		transform: scale(1.5) translateX(2%) translateY(10%) rotate(-12deg);
		outline: 8px solid white;
		opacity: 100%;
		background-color: rgb(245 158 11);
		color: rgb(120 53 15);
	}
</style>
