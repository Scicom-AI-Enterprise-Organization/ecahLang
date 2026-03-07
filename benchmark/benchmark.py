import aiohttp
import asyncio
import time
import json
import click
import os
import statistics

async def stress_test(index, url, model, max_tokens, timeout):
    json_data = {
        'model': model,
        'prompt': 'Hello! ' * 200,
        'max_tokens': max_tokens,
        'stream': True,
        'ignore_eos': True,
    }

    start_time = time.time()
    token_times = []

    client_timeout = aiohttp.ClientTimeout(total=timeout, sock_read=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=json_data) as response:
            first_token_time = None

            async for line in response.content:
                now = time.time()
                elapsed = now - start_time

                l = line.decode()
                if len(l) and 'data: ' in l:
                    try:
                        json.loads(l.split('data: ')[1])
                        token_times.append(now)
                        if first_token_time is None:
                            first_token_time = elapsed
                    except Exception:
                        pass

    total_time = time.time() - start_time
    count = len(token_times)

    # Inter-token latencies
    itl_list = []
    for i in range(1, len(token_times)):
        itl_list.append(token_times[i] - token_times[i - 1])

    return {
        "ttft": first_token_time,
        "total_response": total_time,
        "total_token": count,
        "itl_list": itl_list,
    }

async def run_stress_test(url, model, concurrency, max_tokens, timeout):
    tasks = [
        stress_test(i, url, model, max_tokens, timeout)
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    return results

@click.command()
@click.option('--url', default=None, help='The inference endpoint URL (auto-set from --vllm if not provided).')
@click.option('--model', required=True, help='Model name to use.')
@click.option('--save', required=True, help='save folder name.')
@click.option('--max-tokens', default=384, help='Max tokens to generate per request.')
@click.option('--timeout', default=600, help='Client timeout in seconds per request.')
@click.option('--concurrency-list', default='10,20,30,40,50,60,70,80,90,100',
              help='Comma-separated list of concurrency values to test.')
@click.option('--vllm', is_flag=True, default=False, help='Use vLLM/SGLang endpoint (/v1/completions).')
def main(url, model, save, max_tokens, timeout, concurrency_list, vllm):
    if url is None:
        if vllm:
            url = 'http://localhost:7088/v1/completions'
        else:
            url = 'http://localhost:7088/completions'

    os.makedirs(save, exist_ok=True)
    concurrency_values = [int(c) for c in concurrency_list.split(',')]

    print(f"Endpoint: {url}")
    print(f"Model: {model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Concurrency levels: {concurrency_values}")

    async def run_all():
        for concurrency in concurrency_values:
            print(f"\n{'='*60}")
            print(f"Testing with {concurrency} concurrent requests")
            print(f"{'='*60}")

            bench_start = time.time()
            results = await run_stress_test(url, model, concurrency, max_tokens, timeout)
            bench_duration = time.time() - bench_start

            valid = [r for r in results if r["ttft"] is not None]
            if not valid:
                print("  No successful requests!")
                continue

            # TTFT (Time To First Token)
            ttfts = [r["ttft"] for r in valid]
            avg_ttft = statistics.mean(ttfts)

            # E2E Latency (End-to-End)
            e2e = [r["total_response"] for r in valid]
            avg_e2e = statistics.mean(e2e)

            # ITL (Inter-Token Latency)
            all_itl = []
            for r in valid:
                all_itl.extend(r["itl_list"])
            avg_itl = statistics.mean(all_itl) if all_itl else 0

            # Tokens
            total_tokens = sum(r["total_token"] for r in results)
            avg_tokens = total_tokens / len(results)

            # Throughput
            request_throughput = len(valid) / bench_duration
            output_token_throughput = total_tokens / bench_duration

            results_data = {
                "concurrency": concurrency,
                "successful_requests": len(valid),
                "total_requests": len(results),
                "avg_ttft_s": round(avg_ttft, 4),
                "avg_e2e_latency_s": round(avg_e2e, 4),
                "avg_itl_ms": round(avg_itl * 1000, 4),
                "request_throughput_rps": round(request_throughput, 4),
                "output_token_throughput_tps": round(output_token_throughput, 4),
                "total_tokens": total_tokens,
                "avg_tokens_per_request": round(avg_tokens, 2),
                "bench_duration_s": round(bench_duration, 4),
            }

            print(f"  TTFT:        avg={avg_ttft:.4f}s")
            print(f"  E2E Latency: avg={avg_e2e:.4f}s")
            print(f"  ITL:         avg={avg_itl*1000:.2f}ms")
            print(f"  Throughput:  {request_throughput:.2f} req/s  |  {output_token_throughput:.2f} tok/s")
            print(f"  Tokens:      total={total_tokens}  avg={avg_tokens:.1f}/req")

            with open(f"{save}/{concurrency}concurrency.json", "w") as outfile:
                json.dump(results_data, outfile, indent=4)

    asyncio.run(run_all())

if __name__ == '__main__':
    main()
