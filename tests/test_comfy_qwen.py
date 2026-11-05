import unittest

from tools.image_generator_comfy_qwen import build_workflow, ImageGeneratorComfyQwen, prepare_qwen_prompt


class LocalQwenTests(unittest.TestCase):
    def test_reference_order_and_exact_prompt_reach_native_encoder(self):
        prompt = "Keep the chef from <image1>, use the kitchen from <image2>."
        workflow = build_workflow(prompt, ["chef.png", "kitchen.png"], 42)
        encoder = workflow["4"]["inputs"]
        self.assertEqual(encoder["prompt"], prompt)
        self.assertEqual(workflow[encoder["images.image_1"][0]]["inputs"]["image"], "chef.png")
        self.assertEqual(workflow[encoder["images.image_2"][0]]["inputs"]["image"], "kitchen.png")
        self.assertEqual(workflow["8"]["inputs"]["noise_seed"], 42)

    def test_eight_references_are_supported_and_ninth_is_rejected(self):
        graph = build_workflow("same prompt", [str(i) for i in range(8)], 1)
        self.assertIn("images.image_8", graph["4"]["inputs"])
        with self.assertRaises(ValueError):
            build_workflow("same prompt", [str(i) for i in range(9)], 1)

    def test_cloud_url_rejected_at_backend_construction(self):
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            ImageGeneratorComfyQwen("https://cloud.example")

    def test_standard_uses_base_checkpoint_and_25_step_schedule_with_all_references(self):
        graph = build_workflow("original prompt", [f"ref{i}.png" for i in range(8)], 43,
                               model_variant="standard", steps=25)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "qwen_image_2.1_bf16.safetensors")
        self.assertEqual(graph["10"], {"class_type": "BasicScheduler", "inputs": {
            "model": ["6", 0], "scheduler": "simple", "steps": 25, "denoise": 1.0,
        }})
        self.assertEqual(graph["4"]["inputs"]["prompt"], "original prompt")
        self.assertIn("images.image_8", graph["4"]["inputs"])
        provider = ImageGeneratorComfyQwen("http://127.0.0.1:18184", model_variant="standard")
        self.assertEqual(provider.steps, 25)

    def test_turbo_cannot_silently_use_an_invalid_step_count(self):
        with self.assertRaisesRegex(ValueError, "fixed eight-step"):
            ImageGeneratorComfyQwen("http://127.0.0.1:18184", model_variant="turbo", steps=25)

    def test_reference_labels_map_to_the_actual_qwen_slots(self):
        prompt = "One chef from Image 0 stands in the kitchen from Image 7; Image 1 shows the same chef in profile."
        converted = prepare_qwen_prompt(prompt, 8)
        self.assertIn("One chef from <image1> stands in the kitchen from <image8>", converted)
        self.assertIn("<image2> shows the same chef in profile", converted)
        self.assertNotIn("Image 0", converted)

    def test_native_qwen_labels_are_not_shifted(self):
        self.assertEqual(prepare_qwen_prompt("Use <image1> and <image2>.", 2), "Use <image1> and <image2>.")

    def test_missing_image_slot_fails_before_inference(self):
        for prompt in ["Use Image 2.", "Use <image3>.", "Use <image0>."]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                prepare_qwen_prompt(prompt, 2)

    def test_prompt_formatting_preserves_multi_person_target(self):
        target = "Two adults stand together: the chef on the left and the artist on the right."
        converted = prepare_qwen_prompt(target, 8)
        self.assertEqual(converted, target)
        self.assertNotIn("exactly one person", converted)
