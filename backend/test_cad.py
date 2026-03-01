"""Quick CAD test"""
import asyncio
from cad_agent_v2 import TwoStageCadAgent

async def test():
    def on_thought(text):
        print(text, end='', flush=True)
    
    def on_status(status):
        print(f"\n[STATUS] {status}")
    
    agent = TwoStageCadAgent(
        on_thought=on_thought,
        on_status=on_status
    )
    
    print("=" * 50)
    print("Testing: car")
    print("=" * 50)
    
    result = await agent.generate_prototype("a car")
    
    print("\n" + "=" * 50)
    if result:
        print(f"SUCCESS! STL: {len(result['data'])} bytes")
        print(f"Fallback: {result.get('fallback', False)}")
    else:
        print("FAILED")
    print("=" * 50)
    
    await agent.llm.close()
    return result

if __name__ == "__main__":
    asyncio.run(test())
