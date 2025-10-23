<script lang="ts">
	import Menu from 'lucide-svelte/icons/menu';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { messages, sendMessage } from '$lib/stores/messages';

	let input = '';
	let element;
	export let showInput = true;

	$: if ($messages && element) {
		scrollToBottom(element);
	}

	function sendCommand() {
		if (!input) return;
		const message = { type: 'command', data: { command: input } };
		sendMessage(message);
		input = '';
	}
	function handleKeydown(event) {
		if (event.key === 'Enter') {
			sendCommand();
		}
	}
	const scrollToBottom = async (node) => {
		node.scroll({ top: node.scrollHeight, behavior: 'smooth' });
	};
</script>

<div class=" relative h-full w-full bg-black p-4 text-white">
	<!-- <div class="flex items-center gap-2">
		<span class="text-sm font-semibold">Terminal</span>
	</div> -->
	<div class=" h-full w-full overflow-y-scroll pb-[80px]" bind:this={element}>
		<div class="mt-2 flex flex-col gap-2 font-mono text-xs">
			{#each $messages as message}
				<div
					class="items -center
        flex gap-2"
				>
					<!-- time -->
					<span class="text-gray"
						>{new Date(message.timestamp).toLocaleTimeString([], { timeStyle: 'medium' })}
						{message.botId || ''}
						{message.type}</span
					>
					<!-- show full message object except time and bot -->
					{#if message.data}
						<span>{JSON.stringify(message.data)}</span>
					{:else}
						<span>{JSON.stringify(message)}</span>
					{/if}
				</div>
				<div class="h-50"></div>
			{/each}
		</div>
	</div>
	<!-- terminal input with send -->
	{#if showInput}
		<div class=" absolute bottom-4 mt-2 flex w-[50%] items-center gap-2 bg-black">
			<Input class="w-full" bind:value={input} on:keydown={handleKeydown} />
			<Button on:click={sendCommand} class="bg-muted text-primary" type="submit">Send</Button>
		</div>
	{/if}
</div>
