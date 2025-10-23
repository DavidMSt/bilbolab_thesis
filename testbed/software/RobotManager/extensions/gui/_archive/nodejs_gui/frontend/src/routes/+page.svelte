<script lang="ts">
	import Sidebar from '$lib/components/custom/sidebar.svelte';
	import Terminal from '$lib/components/custom/terminal.svelte';
	import { stream } from '$lib/stores/stream';
	import { currentView } from '$lib/stores/main.js';


	import EmergencyStop from '$lib/components/custom/emergency-stop.svelte';
	import Overview from '$lib/components/custom/overview.svelte';
	import BotDetails from '$lib/components/custom/bot-details.svelte';

	import * as Resizable from '$lib/components/ui/resizable';
	import { onMount } from 'svelte';
	import { initializeWebSocket } from '$lib/stores/messages';

	onMount(() => {
        initializeWebSocket();
    });

</script>

<div
	class="relative grid h-full min-h-screen w-full overflow-hidden md:grid-cols-[220px_1fr] lg:grid-cols-[280px_1fr]"
>
	<Sidebar />

	<main class=" relative h-screen">
		<Resizable.PaneGroup direction="vertical" autoSaveId="pane-oberview-v">
			<Resizable.Pane>
				{#if $currentView=="overview"}
					<Overview/>
				{:else}
					<BotDetails />
				{/if}
			</Resizable.Pane>
			<Resizable.Handle withHandle/>
			<Resizable.Pane defaultSize={20}>
				<Terminal />
			</Resizable.Pane>
		</Resizable.PaneGroup>
	</main>

	<EmergencyStop />
</div>
