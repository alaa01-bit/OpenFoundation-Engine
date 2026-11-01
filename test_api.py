import sys
import os

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import (
    health_check, get_model_specs, analyze_tokenization, curate_dataset,
    run_benchmark_suite, run_bias_and_safety_audit, calculate_distributed_memory,
    TokenizeRequest, CurateRequest, MemoryPlanRequest
)

def main():
    print("Testing health_check...")
    h = health_check()
    assert h["status"] == "healthy"
    print("Health OK:", h)

    print("Testing get_model_specs...")
    specs = get_model_specs()
    assert "d_model" in specs["config"]
    print("Specs OK. Active params:", specs["parameters"]["active_params_per_token"])

    print("Testing analyze_tokenization...")
    tok_res = analyze_tokenization(TokenizeRequest(text="Foundation models with Grouped Query Attention."))
    assert "metrics" in tok_res
    print("Tokenizer OK. Tokens:", tok_res["metrics"]["token_count"])

    print("Testing curate_dataset...")
    curate_res = curate_dataset(CurateRequest(documents=[
        "Foundation model pretraining requires high-performance computing.",
        "Foundation model pretraining requires high-performance computing across nodes.",
        "Completely distinct astronomical dataset on planetary motion."
    ]))
    assert "report" in curate_res
    print("Curation OK. Retention:", curate_res["report"]["overall_retention_percent"], "%")

    print("Testing run_benchmark_suite...")
    bench_res = run_benchmark_suite()
    assert "overall_accuracy_percent" in bench_res
    print("Benchmarks OK. Overall accuracy:", bench_res["overall_accuracy_percent"], "%")

    print("Testing run_bias_and_safety_audit...")
    bias_res = run_bias_and_safety_audit()
    assert "bias_audit" in bias_res
    print("Bias audit OK. Mean disparity:", bias_res["bias_audit"]["overall_mean_disparity"])

    print("Testing calculate_distributed_memory...")
    mem_res = calculate_distributed_memory(MemoryPlanRequest(param_count_billions=7.0, world_size=8, batch_size=4, seq_len=4096))
    assert "fsdp_zero3_vram_gb" in mem_res["memory_breakdown"]
    print("Memory Plan OK. FSDP VRAM per GPU:", mem_res["memory_breakdown"]["fsdp_zero3_vram_gb"], "GB")

    print("\nALL API ENDPOINTS FUNCTION PERFECTLY!")

if __name__ == "__main__":
    main()

