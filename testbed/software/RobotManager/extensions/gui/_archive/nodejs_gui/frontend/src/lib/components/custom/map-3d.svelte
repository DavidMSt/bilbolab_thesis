<script>
	import { onMount } from 'svelte';

	let iframeAvailable = false;

	onMount(async () => {
		try {
			const response = await fetch('http://127.0.0.1:8000', { mode: 'no-cors' });
			// Assuming the fetch doesn't throw, we consider the iframe as available.
			// Note: This is a simplification. In real scenarios, CORS policies might prevent this check from working as intended.
			iframeAvailable = true;
		} catch (error) {
			console.error('Error checking iframe availability:', error);
			iframeAvailable = false;
		}
	});
</script>

<div class="bg-gray h-full w-full">
	{#if iframeAvailable}
		<iframe src="http://127.0.0.1:8000" width="100%" height="100%"></iframe>
	{:else}
        <!-- center error message -->
		<div class="absolute text-neutral-400 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
            <h1 class="text-2xl font-bold">3D Map is not available</h1>
            <p class="text-lg">Please make sure the 3D map server is running.</p>
        </div>
	{/if}
</div>
