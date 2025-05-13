import os
import sys
import time
from typing import Tuple, List, Dict, Optional

# Ensure this matches the CUDA version you compiled llama-cpp-python with AND have installed.
CUDA_VERSION_MAJOR_MINOR = "12.4"
CUDA_TOOLKIT_BASE_DIR = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{CUDA_VERSION_MAJOR_MINOR}"

# Paths to add. Primarily the 'bin' directory.
# Sometimes, other directories like 'libnvvp' might also be needed by some tools,
# but 'bin' is usually the critical one for runtime DLLs like cudart64_XX.dll.
cuda_bin_path = os.path.join(CUDA_TOOLKIT_BASE_DIR, "bin")
# cuda_libnvvp_path = os.path.join(CUDA_TOOLKIT_BASE_DIR, "libnvvp") # Usually not needed for llama_cpp runtime

dll_paths_to_add = []
if os.path.isdir(cuda_bin_path):
    dll_paths_to_add.append(cuda_bin_path)
else:
    print(f"WARNING: CUDA bin directory not found: {cuda_bin_path}", file=sys.stderr)

# if os.path.isdir(cuda_libnvvp_path): # Uncomment if you find it's needed
#     dll_paths_to_add.append(cuda_libnvvp_path)
# else:
#     print(f"WARNING: CUDA libnvvp directory not found: {cuda_libnvvp_path}", file=sys.stderr)

if not dll_paths_to_add:
    print("ERROR: No valid CUDA paths found to add to DLL search path. Exiting.", file=sys.stderr)
    # sys.exit(1) # Or handle more gracefully

for path in dll_paths_to_add:
    try:
        # This is crucial for Python 3.8+ on Windows to find DLLs for C extensions
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(path)
            print(f"Added to DLL search path: {path}")
        else:
            # For older Python or non-Windows, ensure PATH is set (though less effective on modern Windows for this)
            if path not in os.environ['PATH'].split(os.pathsep):
                os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                print(f"Prepended to PATH: {path} (os.add_dll_directory not available)")

    except Exception as e:
        print(f"Error adding {path} to DLL search path: {e}", file=sys.stderr)

# Also, ensure CUDA_HOME / CUDA_PATH are set if any library relies on them directly (though add_dll_directory is better)
if "CUDA_HOME" not in os.environ or os.environ["CUDA_HOME"] != CUDA_TOOLKIT_BASE_DIR:
    os.environ["CUDA_HOME"] = CUDA_TOOLKIT_BASE_DIR
    print(f"Set CUDA_HOME to: {os.environ['CUDA_HOME']}")
if "CUDA_PATH" not in os.environ or os.environ["CUDA_PATH"] != CUDA_TOOLKIT_BASE_DIR:
    os.environ["CUDA_PATH"] = CUDA_TOOLKIT_BASE_DIR
    print(f"Set CUDA_PATH to: {os.environ['CUDA_PATH']}")


# Third-party imports
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma3ChatHandler # Specific handler for Gemma 3 multimodal
except ImportError as e:
    print(f"ERROR: Required llama-cpp-python components not found (Llama, Gemma3ChatHandler): {e}", file=sys.stderr)
    sys.exit(1)


class LLMLoadError: pass


class Prompt_Test:

    _MAX_INFERENCE_ATTEMPTS = 2


    def __init__(self):
        self.llm_path = "C:/_myDrive/repos/auto-vlog/models/GGUF/soob3123_amoral-gemma3-12B-Q4_K_S.gguf"
        self.mmproj_path = "C:/_myDrive/repos/auto-vlog/models/GGUF/mmproj-model-f16-12B.gguf"
        self.n_ctx = 4096
        self.n_gpu_layers = -1
        self.use_flash_atten = True
        self.max_tokens = 1024
        self.temperature = 1.0
        self.top_k=64
        self.min_p=0.01
        self.top_p=0.95
        self.llm = self._load_llm()


    def _load_llm(self) -> Llama:
        """Loads the GGUF LLM and multimodal projector."""
        if not self.llm_path or not self.mmproj_path: raise LLMLoadError("Model paths not set.")
        print(f"Loading LLM ({os.path.basename(self.llm_path)}) & Projector ({os.path.basename(self.mmproj_path)})...")
        try:
            # Gemma3ChatHandler is crucial for enabling multimodal input with Gemma 3 models
            chat_handler = Gemma3ChatHandler(clip_model_path=self.mmproj_path, verbose=True)
            llm = Llama(model_path=self.llm_path, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx,
                        chat_handler=chat_handler, logits_all=False, verbose=True, flash_attn=self.use_flash_atten )
            print(f"LLM loaded with {type(chat_handler).__name__}.")
            return llm
        except Exception as e: raise LLMLoadError(f"Failed to load GGUF model/projector/handler: {e}") from e


    def _execute_single_inference_attempt(
        self,
        messages: List[Dict],
        attempt_num: int,
    ) -> Tuple[Optional[Dict], str, str, bool]:
        """
        Executes one LLM stream attempt, calculates metrics (incl. slow extra call), parses.

        Returns: (parsed_json | None, raw_response_text, usage_string, success_bool)
        """
        print(f"LLM inference attempt {attempt_num}/{self._MAX_INFERENCE_ATTEMPTS}...")
        response_text = ""
        exc = None  # To store potential exceptions
        start_time = time.monotonic()
        elapsed_time = 0.0
        p_tokens, c_tokens, t_tokens = 0, 0, 0 # Prompt, Completion, Total tokens

        try: # Main attempt block
            # Stream LLM Response
            try:
                stream = self.llm.create_chat_completion(
                    messages=messages, 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature, 
                    top_k=self.top_k,
                    min_p=self.min_p, 
                    top_p=self.top_p, 
                    repeat_penalty=1.1,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content')
                    if delta: response_text += delta; print(delta, end='', flush=True)
            except Exception as e_stream: exc = e_stream # Capture stream error
            finally:
                elapsed_time = time.monotonic() - start_time
                if response_text: print()
                if exc: print(f"Stream processing error: {exc}", exc_info=False)

            if exc: return None, "", "", False # Stream failed

            # Calculate Metrics (Post-Stream)
            if response_text:# Completion Tokens
                try: c_tokens = len(self.llm.tokenize(response_text.encode('utf-8')))
                except Exception as e: print(f"Tokenize completion failed: {e}"); c_tokens = 0
            
            print("Skipping num-token statistics for prompt tokens to avoid running LLM a second time.")
            t_tokens = p_tokens + c_tokens # p_tokens will be 0 here

            # TPS & Usage String
            tps_str = ", Speed: N/A"
            if elapsed_time > 0 and c_tokens > 0: tps_str = f", Speed: {(c_tokens / elapsed_time):.2f} tok/s"
            p_tok_display = "N/A" # if p_tokens == 0 else str(p_tokens)
            usage_str = (f"prompt={p_tok_display}, completion={c_tokens}/{self.max_tokens}, "
                            f"total={t_tokens}/{self.n_ctx}, Time: {elapsed_time:.2f}s{tps_str}")
            # Response Validation & Parsing
            if not response_text.strip():
                print(f"Empty LLM response (Attempt {attempt_num}). {usage_str}")
                return None, "", usage_str, False # Failed: Empty response

        except Exception as e: # Catch unexpected errors during the whole attempt
            log_exc = not isinstance(e, ValueError) or "context window" not in str(e).lower()
            print(f"Unexpected error in attempt {attempt_num}: {e}", exc_info=log_exc)
            return None, "", "", False # Failed: Unexpected exception


    def _make_prompt(self):
        messages = []
    
        user_content = []
        user_content.append({"type": "image_url", "image_url": {"url": "C:/_myDrive/repos/auto-vlog/AutoVlogProj/video_candidates/Not today jungle king/judge_frame_00.png"}})
        user_content.append({"type": "text", "text": "describe in detail what you see in the image."})
        messages.append({"role": "user", "content": user_content})
        return messages


if __name__ == "__main__":
    tester = Prompt_Test()
    prompt = tester._make_prompt()
    tester._execute_single_inference_attempt(prompt,1)
    print('complete')
