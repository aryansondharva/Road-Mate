from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import httpx
import base64
from collections import deque
import json
import os
from enum import Enum
from posthog import Posthog  # Note: lowercase posthog
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Posthog configuration
posthog_key = os.getenv("POSTHOG_PROJECT_API_KEY")
if posthog_key:
    posthog = Posthog(posthog_key, host='https://us.i.posthog.com')
else:
    print("Warning: POSTHOG_PROJECT_API_KEY not found in .env. Event tracking disabled.")
    # Dummy capture method to prevent errors later
    class MockPosthog:
        def capture(self, *args, **kwargs): pass
    posthog = MockPosthog()

origins = [
    "*",
    "http://10.76.69.181:8080/",
    "http://localhost",
    "http://localhost:8080",
]

# Configuration from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env. Image analysis will fail.")

MURF_API_KEY = os.getenv("MURF_API_KEY")
if not MURF_API_KEY:
    print("Warning: MURF_API_KEY not found in .env. TTS will be disabled.")

CONTEXT_HISTORY_SIZE = int(os.getenv("CONTEXT_HISTORY_SIZE", "5"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

class CameraType(str, Enum):
    DRIVER = "driver"
    ROAD = "road"

class Frame(BaseModel):
    frame_data: str  # Base64 encoded image data
    camera_type: CameraType
    timestamp: Optional[datetime] = None
    metadata: Optional[dict] = None

class Context(BaseModel):
    timestamp: datetime
    driver_state: Optional[str]
    road_conditions: Optional[str]
    combined_context: str
    frame_ids: Dict[str, str]  # Maps camera_type to frame_id

class ContextGenerator:
    def __init__(self, api_key: str, context_history_size: int = 5):
        self.api_key = api_key
        self.contexts = deque(maxlen=context_history_size)
        self.latest_frames = {}
        
    async def analyze_frame(self, frame_data: str, camera_type: CameraType) -> dict:
        """Analyze a single frame using Gemini API with LLM observability"""
        prompt = (
            "In 1-2 sentences, describe the driver's alertness level, sleep levels and any concerning behaviors. Focus only on key observations."
            if camera_type == CameraType.DRIVER else
            "In 2-3 sentences, summarize the key road conditions and any immediate hazards that need attention."
        )
        
        start_time = time.time()
        trace_id = f"gemini_analysis_{datetime.utcnow().timestamp()}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
                    headers={
                        'Content-Type': 'application/json',
                        'x-goog-api-key': self.api_key,
                    },
                    json={
                        "contents": {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": frame_data
                                    }
                                }
                            ]
                        },
                        "generationConfig": {
                            "temperature": 0.4,
                            "topK": 32,
                            "topP": 1,
                            "maxOutputTokens": 1024,
                        },
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"Gemini API error: {response.text}")
                
                result = response.json()
                latency = time.time() - start_time
                
                # Track LLM generation
                posthog.capture(
                    event="$ai_generation",
                    distinct_id=str(camera_type),
                    properties={
                        "$ai_trace_id": trace_id,
                        "$ai_model": "gemini-2.0-flash",
                        "$ai_provider": "google",
                        "$ai_input": json.dumps([{"role": "user", "content": prompt}]),
                        "$ai_output_choices": json.dumps([{
                            "role": "assistant", 
                            "content": result["candidates"][0]["content"]["parts"][0]["text"]
                        }]),
                        "$ai_latency": latency,
                        "$ai_http_status": response.status_code,
                        "$ai_base_url": "https://generativelanguage.googleapis.com/v1beta",
                        "camera_type": str(camera_type),
                    }
                )
                
                return {
                    "analysis": result["candidates"][0]["content"]["parts"][0]["text"],
                    "timestamp": datetime.utcnow(),
                    "trace_id": trace_id
                }
                
        except Exception as e:
            # Track LLM errors
            posthog.capture(
                event="$ai_generation",
                distinct_id=str(camera_type),
                properties={
                    "$ai_trace_id": trace_id,
                    "$ai_model": "gemini-2.0-flash",
                    "$ai_provider": "google",
                    "$ai_is_error": True,
                    "$ai_error": str(e),
                    "$ai_latency": time.time() - start_time,
                    "camera_type": str(camera_type),
                }
            )
            raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")

    def combine_contexts(self, driver_analysis: Optional[dict], road_analysis: Optional[dict]) -> str:
        """Combine driver and road analyses into a single context"""
        context_parts = []
        
        if driver_analysis:
            context_parts.append(driver_analysis["analysis"])
        if road_analysis:
            context_parts.append(road_analysis["analysis"])
            
        if not context_parts:
            return "No context available"
            
        return " ".join(context_parts)

    async def process_frame(self, frame: Frame) -> Context:
        """Process a new frame and update context"""
        frame_id = f"{frame.camera_type}_{datetime.utcnow().timestamp()}"
        
        # Store the frame analysis
        analysis = await self.analyze_frame(frame.frame_data, frame.camera_type)
        self.latest_frames[frame.camera_type] = {
            "analysis": analysis,
            "frame_id": frame_id
        }
        
        # Generate combined context
        driver_analysis = self.latest_frames.get(CameraType.DRIVER, {}).get("analysis")
        road_analysis = self.latest_frames.get(CameraType.ROAD, {}).get("analysis")
        
        context = Context(
            timestamp=datetime.utcnow(),
            driver_state=driver_analysis["analysis"] if driver_analysis else None,
            road_conditions=road_analysis["analysis"] if road_analysis else None,
            combined_context=self.combine_contexts(driver_analysis, road_analysis),
            frame_ids={
                camera_type: data["frame_id"]
                for camera_type, data in self.latest_frames.items()
            }
        )
        
        self.contexts.append(context)
        return context

    def get_latest_context(self) -> Optional[Context]:
        """Get the most recent context"""
        return self.contexts[-1] if self.contexts else None

    def get_context_history(self) -> List[Context]:
        """Get all stored contexts"""
        return list(self.contexts)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize context generator
context_generator = ContextGenerator(
    api_key=GOOGLE_API_KEY,
    context_history_size=CONTEXT_HISTORY_SIZE
)


@app.middleware("http")
async def track_timing(request: Request, call_next):
    """Middleware to track request timing"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = round((time.time() - start_time) * 1000, 2)
    
    if request.url.path != "/health":  # Don't track health checks
        posthog.capture(
            distinct_id=str(datetime.utcnow().timestamp()),
            event='api_request',
            properties={
                'path': request.url.path,
                'method': request.method,
                'duration_ms': duration,
                'status_code': response.status_code
            }
        )
    
    return response


@app.post("/frames/")
async def process_frame(frame: Frame):
    """Process incoming frame with PostHog tracking"""
    try:
        start_time = time.time()
        
        # Generate a unique ID for this analysis session
        analysis_id = f"{frame.camera_type}_{datetime.utcnow().timestamp()}"
        
        # Track frame submission
        posthog.capture(
            distinct_id=analysis_id,
            event='frame_submitted',
            properties={
                'camera_type': frame.camera_type,
                'has_metadata': bool(frame.metadata),
                'frame_size': len(frame.frame_data)
            }
        )
        
        # Process the frame and get analysis
        analysis_result = await context_generator.process_frame(frame)
        
        # Track successful analysis
        posthog.capture(
            distinct_id=analysis_id,
            event='frame_analyzed',
            properties={
                'camera_type': frame.camera_type,
                'total_duration_ms': round((time.time() - start_time) * 1000, 2)
            }
        )
        
        # Return the actual analysis result
        return {
            "timestamp": analysis_result.timestamp.isoformat(),
            "driver_state": analysis_result.driver_state,
            "road_conditions": analysis_result.road_conditions,
            "combined_context": analysis_result.combined_context,
            "frame_ids": analysis_result.frame_ids,
            "analysis_id": analysis_id  # Include the ID if you need it
        }
    
    except Exception as e:
        # Track errors
        posthog.capture(
            distinct_id=str(datetime.utcnow().timestamp()),
            event='analysis_error',
            properties={
                'camera_type': frame.camera_type,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        )
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/frames/", response_model=Context)
# async def process_frame(frame: Frame):
#     return await context_generator.process_frame(frame)

@app.get("/context/latest", response_model=Optional[Context])
async def get_latest_context():
    return context_generator.get_latest_context()

@app.get("/context/history", response_model=List[Context])
async def get_context_history():
    return context_generator.get_context_history()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "en-US-cooper"  # Default Murf Falcon voice
    speed: Optional[float] = 1.0


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Murf Falcon TTS API"""
    if not MURF_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Murf API key not configured. Set MURF_API_KEY in environment variables."
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://global.api.murf.ai/v1/speech/stream",
                headers={
                    "api-key": MURF_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "voiceId": request.voice_id,
                    "text": request.text,
                    "modelVersion": "FALCON",
                    "speed": request.speed,
                    "format": "MP3",
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Murf API error: {response.text}"
                )

            # Track TTS usage
            posthog.capture(
                distinct_id=str(datetime.utcnow().timestamp()),
                event="tts_generated",
                properties={
                    "voice_id": request.voice_id,
                    "text_length": len(request.text),
                    "model": "FALCON"
                }
            )

            from fastapi.responses import Response
            return Response(
                content=response.content,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=speech.mp3"}
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.delete("/context/clear")
async def clear_context():
    """Clear all stored contexts and frame analysis"""
    try:
        context_generator.contexts.clear()
        context_generator.latest_frames.clear()
        
        # Track context clearing in PostHog
        posthog.capture(
            distinct_id=str(datetime.utcnow().timestamp()),
            event='context_cleared',
            properties={
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        return {"status": "success", "message": "Context history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear context: {str(e)}")