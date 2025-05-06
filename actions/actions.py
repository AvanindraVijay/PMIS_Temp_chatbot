import requests
import logging
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import EventType


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionGreet(Action):
    def name(self) -> Text:
        return "action_greet"
    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[EventType]:
        dispatcher.utter_message(text="Hello! How can I help you today?")
        return []


class ActionGoodbye(Action):
    def name(self) -> Text:
        return "action_goodbye"
    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[EventType]:
        dispatcher.utter_message(text="Goodbye! If you have any more questions, feel free to ask.")
        return []

class ActionRefundPolicy(Action):
    def name(self) -> Text:
        return "action_refund_policy"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[EventType]:
        logger.info("Refund policy action triggered")
        dispatcher.utter_message(text=(
            "We're sorry, but we currently operate a no-refund policy. "
            "Please refer to our terms and conditions for more details."
        ))
        return []
    
    
class ActionLlamaFallback(Action):
    def name(self) -> str:
        return "action_llama_fallback"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[EventType]:
        last_user_message = tracker.latest_message.get("text")
        

        logger.info(f"LLaMA fallback triggered for: {last_user_message}")
        

        dispatcher.utter_message(text="Let me find information about that for you...")
        

        try:
            logger.info(f"Sending to LLaMA API: {last_user_message}")
            response = requests.post("http://localhost:8000/chat", 
                                    json={"message": last_user_message}, 
                                    timeout=90)
            logger.info(f"LLaMA API response status: {response.status_code}")
            if response.status_code == 200:
                llama_reply = response.json().get("response", "Sorry, I couldn't find specific information about that.")
                dispatcher.utter_message(text=llama_reply)
            else:
                logger.error(f"LLaMA API error: {response.text}")
                dispatcher.utter_message(text="Sorry, I'm having trouble retrieving that information right now.")
        except Exception as e:
            logger.error(f"Error connecting to LLaMA API: {str(e)}")
            dispatcher.utter_message(text="I apologize, but I'm unable to access that information at the moment. Please try again later.")
        return []