#!/usr/bin/env python3
"""
LLM-powered variable generation for prompt templates.

This module uses a small LLM to generate creative values for template variables
instead of using fixed predefined lists.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from ..core.device import get_device_and_dtype


@dataclass
class PromptVariable:
    """Represents a variable in a prompt template."""

    name: str
    values: List[str]
    description: Optional[str] = None
    llm_generated: bool = False


class LLMVariableGenerator:
    """
    Uses a small LLM to generate creative values for prompt variables.

    This allows for dynamic, context-aware variable generation instead of
    fixed predefined lists.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_api: bool = True):
        """
        Initialize LLM variable generator using instruction-following model.

        Args:
            model_name: Name of Hugging Face model to use (default: Qwen/Qwen2.5-1.5B-Instruct)
            use_api: Whether to try API-based LLMs first (OpenAI, Anthropic)
        """
        self.model_name = model_name
        self.use_api = use_api
        # Auto-detect optimal device
        self.device, self.dtype = get_device_and_dtype()
        self.provider = "rulebased"  # Start with simple rule-based generation

        # Try API-based LLMs if enabled
        if use_api:
            self._setup_api_model()

        # Use local model as absolute last resort (often has compatibility issues)
        # For now, stick with rule-based which works reliably
        print("✅ Using rule-based semantic generation")
        print("   Works great for common variables like colors, backgrounds, styles")

    def _setup_api_model(self):
        """Setup API-based generation using ConceptNet or OpenAI."""
        # Try ConceptNet first (free, fast, perfect for this use case)
        try:
            import requests
            # Test ConceptNet API
            response = requests.get("http://api.conceptnet.io/c/en/hair", timeout=2)
            if response.status_code == 200:
                self.provider = "conceptnet"
                print("✅ Using ConceptNet for semantic variable generation")
                print("   Free knowledge graph - perfect for generating related concepts")
                return
        except Exception:
            pass  # ConceptNet not available, try OpenAI

        # Fall back to OpenAI if available
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=api_key)
                    self.provider = "openai"
                    print("✅ Using OpenAI API for variable generation")
                    return
                except ImportError:
                    print("💡 Install openai package: pip install openai")
        except Exception as e:
            pass

    def _generate_from_conceptnet(self, variable_name: str, num_values: int = 10) -> List[str]:
        """Generate values using ConceptNet semantic knowledge graph."""
        import requests
        import re

        # Convert variable name to natural language (hair_color → "hair color")
        concept = variable_name.replace("_", " ")

        values = []
        seen = set()

        # Strategy 1: Query for things that have this property
        # e.g., "What colors can hair be?"
        try:
            url = f"http://api.conceptnet.io/query?node=/c/en/{variable_name.replace('_', '_')}&rel=/r/HasProperty&limit=50"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for edge in data.get('edges', []):
                    # Extract the related concept
                    end = edge.get('end', {}).get('label', '')
                    if end and end.lower() not in seen:
                        # Clean up the value
                        value = end.strip().title()
                        if len(value) > 2 and len(value) < 30:
                            values.append(value)
                            seen.add(end.lower())
        except Exception as e:
            print(f"   ConceptNet query 1 failed: {e}")

        # Strategy 2: Search for the concept itself and get related terms
        try:
            # For "hair_color", search for both "hair" and "color" concepts
            parts = variable_name.split("_")
            for part in parts:
                url = f"http://api.conceptnet.io/query?node=/c/en/{part}&limit=50"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    for edge in data.get('edges', []):
                        # Look for RelatedTo, IsA, HasProperty relationships
                        rel = edge.get('rel', {}).get('label', '')
                        if rel in ['RelatedTo', 'IsA', 'HasProperty', 'AtLocation']:
                            # Get both start and end nodes
                            for node_key in ['start', 'end']:
                                label = edge.get(node_key, {}).get('label', '')
                                if label and label.lower() not in seen and part in label.lower():
                                    value = label.strip().title()
                                    if len(value) > 2 and len(value) < 30 and value.lower() != part:
                                        values.append(value)
                                        seen.add(label.lower())
        except Exception as e:
            print(f"   ConceptNet query 2 failed: {e}")

        # Strategy 3: If it's a color/appearance variable, add common values
        if 'color' in variable_name.lower():
            common_colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange',
                           'Pink', 'Black', 'White', 'Brown', 'Grey', 'Silver', 'Gold']
            for color in common_colors:
                if color.lower() not in seen:
                    values.append(color)
                    seen.add(color.lower())

        # Fallback: Generate from variable name itself
        if len(values) < num_values:
            print(f"   ⚠️  ConceptNet returned {len(values)} values, using fallback generation")
            # Use rule-based generation for common variable types
            fallback = self._generate_simple_values(variable_name, num_values)
            for val in fallback:
                if val.lower() not in seen:
                    values.append(val)
                    seen.add(val.lower())

        result = values[:num_values]
        print(f"   ✓ Generated {len(result)} values from ConceptNet")
        return result

    def _generate_simple_values(self, variable_name: str, num_values: int) -> List[str]:
        """Simple rule-based generation for common variable types."""
        # Common patterns
        if 'color' in variable_name.lower():
            return ['Blonde', 'Brunette', 'Red', 'Black', 'Auburn', 'Platinum', 'Silver', 'Grey', 'Brown', 'Copper']
        elif 'ethnic' in variable_name.lower() or 'background' in variable_name.lower():
            return ['Asian', 'European', 'African', 'Hispanic', 'Middle Eastern', 'Pacific Islander',
                   'Native American', 'South Asian', 'Caribbean', 'Mediterranean']
        elif 'style' in variable_name.lower():
            return ['Casual', 'Formal', 'Elegant', 'Sporty', 'Vintage', 'Modern', 'Classic', 'Bohemian', 'Minimalist', 'Eclectic']
        else:
            # Generic placeholder
            return [f"Value {i+1}" for i in range(num_values)]

    def _generate_from_openai(self, variable_name: str, context: str, num_values: int) -> List[str]:
        """Generate values using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate diverse, creative values for prompt template variables. Output only a JSON array of values."},
                    {"role": "user", "content": f"Generate {num_values} diverse values for the variable '{variable_name}' in the context: '{context}'. Output format: {{\"values\": [\"value1\", \"value2\", ...]}}"}
                ],
                temperature=0.8,
            )

            import json
            result = json.loads(response.choices[0].message.content)
            values = result.get('values', [])
            print(f"   ✓ Generated {len(values)} values from OpenAI")
            return values[:num_values]
        except Exception as e:
            print(f"   ❌ OpenAI generation failed: {e}")
            return self._generate_simple_values(variable_name, num_values)

    def _setup_huggingface_model(self):
        """Setup instruction-following model for value generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"🤖 Loading LLM: {self.model_name}")

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # Set pad token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model with optimal settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    device_map=self.device if self.device in ["cuda", "mps"] else None,
                    trust_remote_code=True,
                    attn_implementation="eager",  # Use eager attention (no flash-attention required)
                )

                # Move to device if not using device_map
                if self.device == "cpu":
                    self.model = self.model.to(self.device)

                self.provider = "huggingface"
                print(f"✅ Successfully loaded {self.model_name}")
                print(f"   Ready for creative value generation")

            except Exception as e:
                print(f"❌ Failed to load {self.model_name}: {e}")
                print(f"   Use --define to manually specify variable values")
                self.provider = "failed"

        except ImportError as e:
            print(f"❌ Transformers not available: {e}")
            print(f"   Install with: pip install transformers")
            self.provider = "failed"

    def generate_variable_values(
        self, variable_name: str, context: str, num_values: int = 10
    ) -> List[str]:
        """
        Generate values for a variable using ConceptNet, OpenAI, or local model.

        Args:
            variable_name: Name of the variable (e.g., "hair_color", "ethnicity")
            context: The full prompt template for context
            num_values: Number of values to generate

        Returns:
            List of generated values
        """
        if self.provider == "conceptnet":
            return self._generate_from_conceptnet(variable_name, num_values)
        elif self.provider == "openai":
            return self._generate_from_openai(variable_name, context, num_values)
        elif self.provider == "rulebased":
            print(f"   Generating values for: {variable_name}")
            return self._generate_simple_values(variable_name, num_values)
        elif self.provider == "huggingface":
            try:
                import torch

                # Create a clear, instruction-based prompt
                prompt = f"""You are a creative assistant. Generate {num_values} diverse and creative values for the variable "{variable_name}".

Context: This will be used in a prompt template: "{context}"

Requirements:
- Generate exactly {num_values} different values
- Make them diverse and creative
- Output ONLY as a JSON list in this exact format: {{"values": ["value1", "value2", "value3"]}}
- Do not include any other text before or after the JSON

JSON output:"""

                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate with greedy decoding (more stable on MPS with float16)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=400,
                        do_sample=False,  # Use greedy decoding to avoid float16 precision issues
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Debug: show what the model generated
                print(f"\n   Full model output:\n{generated_text}\n")

                # Extract the JSON part (after the prompt)
                if "JSON output:" in generated_text:
                    json_part = generated_text.split("JSON output:")[-1].strip()
                else:
                    json_part = generated_text

                # Try to parse as JSON
                try:
                    import json as json_module
                    # Find JSON object in text
                    start_idx = json_part.find("{")
                    end_idx = json_part.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = json_part[start_idx:end_idx]
                        data = json_module.loads(json_str)
                        if "values" in data and isinstance(data["values"], list):
                            values = [str(v).strip() for v in data["values"] if v]
                            print(f"   ✓ Parsed JSON, found {len(values)} values")
                            return values[:num_values]
                except Exception as e:
                    print(f"   ⚠️  JSON parsing failed: {e}")

                # Fallback: try to extract comma-separated values
                print(f"   Trying fallback extraction...")
                # Remove JSON artifacts including brackets
                cleaned = json_part.replace("{", "").replace("}", "").replace('"', "").replace("'", "")
                cleaned = cleaned.replace("[", "").replace("]", "")  # Remove array brackets
                if ":" in cleaned:
                    cleaned = cleaned.split(":")[-1]

                values = [v.strip() for v in cleaned.split(",") if v.strip()]
                if values:
                    print(f"   ✓ Extracted {len(values)} values via fallback")
                return values[:num_values] if values else []

            except Exception as e:
                print(f"❌ Error generating with LLM: {e}")
                import traceback
                traceback.print_exc()
                return self._generate_rule_based(variable_name, context, num_values)
        else:
            # Model failed to load
            return self._generate_rule_based(variable_name, context, num_values)

    def _generate_rule_based(
        self, variable_name: str, context: str, num_values: int = 10
    ) -> List[str]:
        """
        Generate variable values using rule-based approach (fallback).

        Args:
            variable_name: Name of the variable
            context: The full prompt template for context
            num_values: Number of values to generate

        Returns:
            List of generated values
        """
        # LLM generation failed - recommend using --define
        print(f"\n❌ Could not generate values for '{variable_name}'")
        print(f"💡 Recommendation: Specify custom values with --define:")
        print(f"   --define '{variable_name}=value1,value2,value3'")
        print(f"\n💡 Or use an API-based LLM (OpenAI, Anthropic) for better results")
        return []


class SmartPromptTemplate:
    """
    Enhanced prompt template that uses LLM to generate variable values.

    Automatically detects variables in templates and generates appropriate values
    using LLM intelligence for more creative and context-aware results.
    """

    def __init__(
        self,
        template: str,
        variable_generator: Optional[LLMVariableGenerator] = None,
        custom_variables: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize smart prompt template.

        Args:
            template: Template string with {variable} placeholders
            variable_generator: LLM generator for creating variable values
            custom_variables: Pre-defined variables (overrides LLM generation)
        """
        self.template = template
        self.variable_generator = variable_generator or LLMVariableGenerator()
        self.custom_variables = custom_variables or {}
        self.variables = {}
        self._detect_and_generate_variables()

    def _detect_and_generate_variables(self):
        """Detect variables in template and generate values."""
        # Find all {variable} placeholders in template
        pattern = r"\{([^}]+)\}"
        template_vars = re.findall(pattern, self.template)

        for var_name in template_vars:
            if var_name in self.custom_variables:
                # Use provided custom values
                self.variables[var_name] = self.custom_variables[var_name]
            else:
                # Generate values using LLM
                print(f"🤖 Generating values for variable: {var_name}")
                values = self.variable_generator.generate_variable_values(
                    var_name, self.template, num_values=10
                )
                if not values:
                    print(f"❌ No values generated for '{var_name}' - use --define to specify manually")
                    self.variables[var_name] = []
                else:
                    self.variables[var_name] = values
                    print(f"✅ Generated {len(values)} values for {var_name}")

    def generate_prompts(self, max_combinations: Optional[int] = None) -> List[str]:
        """
        Generate all possible prompt combinations from template.

        Args:
            max_combinations: Maximum number of prompts to generate (None for all)

        Returns:
            List of expanded prompt strings
        """
        if not self.variables:
            return [self.template]

        # Check if any variable has no values
        empty_vars = [name for name, values in self.variables.items() if not values]
        if empty_vars:
            print(f"\n❌ Cannot generate prompts - these variables have no values: {', '.join(empty_vars)}")
            print(f"   Use --define to specify values for each variable")
            return []

        # Generate all combinations using Cartesian product with randomization
        import itertools
        import random

        var_names = list(self.variables.keys())
        var_values = [self.variables[name] for name in var_names]

        prompts = []
        all_combinations = list(itertools.product(*var_values))

        # Shuffle combinations for better diversity
        random.shuffle(all_combinations)

        for combination in all_combinations:
            # Create mapping for this combination
            var_mapping = dict(zip(var_names, combination))

            # Format template with this combination
            prompt = self.template.format(**var_mapping)
            prompts.append(prompt)

            # Stop if we've reached max combinations
            if max_combinations and len(prompts) >= max_combinations:
                break

        return prompts

    def get_total_combinations(self) -> int:
        """Calculate total number of possible prompt combinations."""
        if not self.variables:
            return 1

        total = 1
        for values in self.variables.values():
            total *= len(values)
        return total

    def regenerate_variable(self, variable_name: str, num_values: int = 10):
        """Regenerate values for a specific variable."""
        if variable_name in self.variables:
            print(f"🔄 Regenerating values for: {variable_name}")
            values = self.variable_generator.generate_variable_values(
                variable_name, self.template, num_values
            )
            self.variables[variable_name] = values
            print(f"✅ Regenerated {len(values)} values for {variable_name}")
        else:
            print(f"❌ Variable '{variable_name}' not found in template")

    def __str__(self) -> str:
        """String representation of template."""
        return f"SmartPromptTemplate('{self.template}') with {len(self.variables)} variables"


def create_smart_template(
    template: str,
    custom_variables: Optional[Dict[str, List[str]]] = None,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> SmartPromptTemplate:
    """
    Create a smart prompt template with LLM-powered variable generation.

    Args:
        template: Template string with {variable} placeholders
        custom_variables: Pre-defined variables (overrides LLM generation)
        model_name: Hugging Face model to use (default: Qwen/Qwen2.5-1.5B-Instruct)

    Returns:
        SmartPromptTemplate instance
    """
    generator = LLMVariableGenerator(model_name=model_name)
    return SmartPromptTemplate(template, generator, custom_variables)
