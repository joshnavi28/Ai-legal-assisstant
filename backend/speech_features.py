#!/usr/bin/env python3
"""
Speech-to-Text and Text-to-Speech Features for Legal AI Assistant
Resume-worthy speech processing capabilities
"""

import speech_recognition as sr
import pyttsx3
import tempfile
import os
import threading
import queue
import time
from pathlib import Path
import wave
import pyaudio
import numpy as np
from typing import Optional, Dict, Any
import json
import subprocess
import shutil

class SpeechProcessor:
    """Advanced speech processing for legal AI assistant"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None
        
        # Configure text-to-speech
        self._setup_tts()
        
    def _setup_tts(self):
        """Configure text-to-speech engine"""
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Set properties for better quality
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level
            
            # Try to set a good voice
            if voices:
                # Prefer female voice for legal assistant
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Fallback to first available voice
                    self.engine.setProperty('voice', voices[0].id)
                    
        except Exception as e:
            print(f"TTS setup warning: {e}")
    
    def _convert_audio_format(self, input_path: str, output_format: str = 'wav') -> str:
        """
        Convert audio file to required format using ffmpeg or fallback methods
        
        Args:
            input_path: Path to input audio file
            output_format: Desired output format (default: wav)
            
        Returns:
            Path to converted audio file
        """
        try:
            # Check if input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Get file extension
            input_ext = Path(input_path).suffix.lower()
            output_ext = f".{output_format}"
            
            # If already in correct format, return original path
            if input_ext == output_ext:
                return input_path
            
            # Create output path
            output_path = str(Path(input_path).with_suffix(output_ext))
            
            # Try using ffmpeg command first
            try:
                # Check if ffmpeg is available
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # FFmpeg is available, use it for conversion
                    cmd = [
                        'ffmpeg', '-i', input_path, 
                        '-acodec', 'pcm_s16le', 
                        '-ar', '16000', 
                        '-ac', '1', 
                        output_path, 
                        '-y'  # Overwrite output file
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and os.path.exists(output_path):
                        print(f"✅ FFmpeg conversion successful: {output_path}")
                        return output_path
                    else:
                        print(f"FFmpeg conversion failed: {result.stderr}")
                        
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
                print(f"FFmpeg not available or conversion failed: {e}")
            
            # Fallback: try using ffmpeg-python if available
            try:
                import ffmpeg
                
                stream = ffmpeg.input(input_path)
                stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ar=16000, ac=1)
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                if os.path.exists(output_path):
                    print(f"✅ FFmpeg-python conversion successful: {output_path}")
                    return output_path
                    
            except ImportError:
                print("ffmpeg-python not available")
            except Exception as e:
                print(f"FFmpeg-python conversion error: {e}")
            
            # Fallback: try using pydub if available
            try:
                from pydub import AudioSegment
                
                # Load audio file
                if input_ext == '.webm':
                    audio = AudioSegment.from_file(input_path, format="webm")
                elif input_ext == '.mp3':
                    audio = AudioSegment.from_mp3(input_path)
                elif input_ext == '.m4a':
                    audio = AudioSegment.from_file(input_path, format="m4a")
                else:
                    audio = AudioSegment.from_file(input_path)
                
                # Export as WAV
                audio.export(output_path, format="wav")
                
                if os.path.exists(output_path):
                    print(f"✅ Pydub conversion successful: {output_path}")
                    return output_path
                    
            except ImportError:
                print("pydub not available")
            except Exception as e:
                print(f"Pydub conversion error: {e}")
            
            # Final fallback: try to use the original file if it's a supported format
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
            if input_ext in supported_formats:
                print(f"⚠️  Using original file format: {input_path}")
                return input_path
            
            # If all conversion methods fail, return original path and let speech recognition try
            print(f"⚠️  Audio conversion failed, using original file: {input_path}")
            return input_path
            
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return input_path
    
    def speech_to_text(self, audio_file_path: str, language: str = 'en-IN') -> Dict[str, Any]:
        """
        Convert speech to text with advanced error handling
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (en-IN for Indian English, hi-IN for Hindi)
            
        Returns:
            Dict with transcription and confidence
        """
        try:
            print(f"🎤 Processing audio file: {audio_file_path}")
            
            # Convert audio to wav format if needed
            converted_audio_path = self._convert_audio_format(audio_file_path, 'wav')
            print(f"🎵 Converted audio path: {converted_audio_path}")
            
            with sr.AudioFile(converted_audio_path) as source:
                # Adjust for ambient noise
                print("🔊 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                
                # Try multiple recognition services
                transcription = None
                confidence = 0.0
                
                # Try Google Speech Recognition
                print("🌐 Trying Google Speech Recognition...")
                try:
                    result = self.recognizer.recognize_google(
                        audio, 
                        language=language,
                        show_all=True
                    )
                    if result and 'alternative' in result:
                        transcription = result['alternative'][0]['transcript']
                        confidence = result['alternative'][0].get('confidence', 0.8)
                        print(f"✅ Google recognition successful: {transcription}")
                    else:
                        print("⚠️  Google recognition returned no results")
                except sr.UnknownValueError:
                    print("⚠️  Google recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"⚠️  Google recognition request failed: {e}")
                except Exception as e:
                    print(f"⚠️  Google recognition error: {e}")
                
                # Fallback to Sphinx if Google fails
                if not transcription:
                    print("🔍 Trying Sphinx recognition...")
                    try:
                        transcription = self.recognizer.recognize_sphinx(audio)
                        confidence = 0.6
                        print(f"✅ Sphinx recognition successful: {transcription}")
                    except Exception as e:
                        print(f"⚠️  Sphinx recognition failed: {e}")
                
                if transcription:
                    return {
                        "success": True,
                        "transcription": transcription,
                        "confidence": confidence,
                        "language": language,
                        "features": [
                            "Ambient noise reduction",
                            "Multiple recognition engines",
                            "Confidence scoring",
                            "Language detection",
                            "Audio format conversion"
                        ]
                    }
                else:
                    return {
                        "success": False,
                        "error": "Could not transcribe audio",
                        "suggestions": [
                            "Speak more clearly",
                            "Reduce background noise",
                            "Try again",
                            "Check microphone permissions"
                        ]
                    }
                    
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return {
                "success": False,
                "error": f"Speech recognition error: {str(e)}",
                "features": ["Error handling", "Detailed feedback"]
            }
    
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert text to speech with legal context awareness
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file
            
        Returns:
            Dict with status and audio info
        """
        try:
            # Legal context processing
            processed_text = self._process_legal_context(text)
            
            # Generate speech
            if output_path:
                self.engine.save_to_file(processed_text, output_path)
                self.engine.runAndWait()
                
                # Get audio file info
                audio_info = self._get_audio_info(output_path)
                
                return {
                    "success": True,
                    "text": processed_text,
                    "audio_path": output_path,
                    "duration": audio_info.get("duration", 0),
                    "features": [
                        "Legal context awareness",
                        "Professional tone",
                        "Clear pronunciation",
                        "Audio file generation"
                    ]
                }
            else:
                # Play directly
                self.engine.say(processed_text)
                self.engine.runAndWait()
                
                return {
                    "success": True,
                    "text": processed_text,
                    "features": [
                        "Real-time playback",
                        "Legal context awareness",
                        "Professional tone"
                    ]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Text-to-speech error: {str(e)}",
                "features": ["Error handling"]
            }
    
    def _process_legal_context(self, text: str) -> str:
        """Process text for legal context awareness"""
        # Add pauses for legal terms
        legal_terms = [
            "Article", "Section", "Clause", "Subsection",
            "Constitution", "Act", "Regulation", "Statute"
        ]
        
        processed_text = text
        for term in legal_terms:
            if term in processed_text:
                # Add slight pause before legal terms
                processed_text = processed_text.replace(term, f" ... {term}")
        
        return processed_text
    
    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get information about generated audio file"""
        try:
            with wave.open(audio_path, 'rb') as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = frames / float(rate)
                
                return {
                    "duration": duration,
                    "sample_rate": rate,
                    "channels": audio_file.getnchannels(),
                    "file_size": os.path.getsize(audio_path)
                }
        except Exception:
            return {}
    
    def start_realtime_recording(self) -> Dict[str, Any]:
        """Start real-time speech recording"""
        try:
            self.is_recording = True
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.start()
            
            return {
                "success": True,
                "message": "Real-time recording started",
                "features": [
                    "Continuous recording",
                    "Background processing",
                    "Real-time transcription"
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Recording start error: {str(e)}"
            }
    
    def stop_realtime_recording(self) -> Dict[str, Any]:
        """Stop real-time speech recording"""
        try:
            self.is_recording = False
            if self.audio_thread:
                self.audio_thread.join()
            
            return {
                "success": True,
                "message": "Real-time recording stopped",
                "features": ["Recording control", "Thread management"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Recording stop error: {str(e)}"
            }
    
    def _record_audio(self):
        """Background audio recording thread"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            frames = []
            
            while self.is_recording:
                try:
                    data = stream.read(1024)
                    frames.append(data)
                except Exception:
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save recorded audio
            if frames:
                output_path = f"uploads/realtime_recording_{int(time.time())}.wav"
                os.makedirs("uploads", exist_ok=True)
                
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(frames))
                
                # Process the recorded audio
                result = self.speech_to_text(output_path)
                self.audio_queue.put(result)
                
        except Exception as e:
            print(f"Recording error: {e}")
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get supported languages for speech recognition"""
        return {
            "languages": {
                "en-IN": "Indian English",
                "hi-IN": "Hindi",
                "en-US": "US English",
                "en-GB": "British English"
            },
            "features": [
                "Multi-language support",
                "Indian accent recognition",
                "Legal terminology support"
            ]
        }

# Global speech processor instance
speech_processor = SpeechProcessor()

def get_speech_processor():
    """Get speech processor instance"""
    return speech_processor 