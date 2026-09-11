import sys
import os
import asyncio
sys.path.insert(0, 'k:/Hiver_assignment/Customer_Support_AI_Agent/backend')

from app.api.main import lifespan, app, generate_reply, GenerateReplyRequest

async def run_test():
    async with lifespan(app):
        req = GenerateReplyRequest(message="My app is crashing", conversation_context="")
        res = await generate_reply(req)
        print("Intent:", res.intent)
        print("Confidence:", res.intent_confidence)
        print("Retrieved Cases:", len(res.retrieved_cases))
        print("Reply:", res.reply[:100])
        print("Should Escalate:", res.should_escalate)

asyncio.run(run_test())
