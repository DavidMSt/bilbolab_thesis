<script>
	import * as Select from '$lib/components/ui/select/index.js';
	import Label from '../ui/label/label.svelte';
</script>

<h2 class="mb-4 text-lg font-semibold md:text-2xl">Parameters</h2>


<Label>Controller</Label>
<Select.Root>
	<Select.Trigger class="w-full">
		<Select.Value placeholder="Select Controller" />
	</Select.Trigger>
	<Select.Content>
		<Select.Item value="Velocity">Velocity</Select.Item>
		<Select.Item value="Position">Position</Select.Item>
		<Select.Item value="Torque">Torque</Select.Item>
	</Select.Content>
</Select.Root>
