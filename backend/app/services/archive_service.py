import asyncio
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from google import genai
from google.genai import types
from fastapi import UploadFile, HTTPException
from supabase import StorageException

from app.core.config import settings
from app.core.supabase import get_supabase_client
from app.schemas.archive import ArchiveResponse


class ArchiveService:
    """
    Service for handling archive operations including file upload,
    content analysis, and embedding generation using Google GenAI.
    """
    
    def __init__(self):
        """Initialize Google GenAI client."""
        self._client = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.model = "gemini-2.5-flash-lite"
        self.embedding_model = "text-embedding-004"
        self.supabase_client = get_supabase_client()
        if not settings.SUPABASE_STORAGE_BUCKET:
            raise ValueError("SUPABASE_STORAGE_BUCKET is not configured")
        self.storage_bucket = settings.SUPABASE_STORAGE_BUCKET
    
    @property
    def client(self):
        """Lazy initialization of Google GenAI client."""
        if self._client is None:
            if not settings.GOOGLE_GENAI_API_KEY:
                raise ValueError("GOOGLE_GENAI_API_KEY is not configured")
            self._client = genai.Client(api_key=settings.GOOGLE_GENAI_API_KEY)
        return self._client
    
    def _create_temp_file(self, content: bytes, suffix: str) -> str:
        """Persist bytes to a temporary file and return its path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            return tmp_file.name
    
    def _cleanup_temp_file(self, file_path: str):
        """Remove temporary file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            # Log error but don't fail
            print(f"Error cleaning up temp file {file_path}: {e}")
    
    def _build_storage_path(self, filename: Optional[str]) -> str:
        """Generate a safe storage path for Supabase storage."""
        base_name = filename or "uploaded_file"
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
        return f"archives/{uuid4().hex}/{safe_name}"

    async def _upload_file_to_supabase_storage(
        self,
        storage_path: str,
        content: bytes,
        mime_type: Optional[str],
    ) -> str:
        """Upload file content to Supabase Storage and return the storage path."""
        storage_bucket = self.supabase_client.storage.from_(self.storage_bucket)
        loop = asyncio.get_event_loop()

        def _upload():
            storage_bucket.upload(
                storage_path,
                content,
                file_options={
                    "content-type": mime_type or "application/octet-stream",
                },
            )
            return storage_path

        try:
            return await loop.run_in_executor(self._executor, _upload)
        except StorageException as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file to storage: {exc.message}",
            ) from exc
    
    def _get_comprehensive_analysis_prompt(
        self,
        title: str,
        media_types: List[str],
        tags: List[str],
        description: str
    ) -> str:
        """
        Generate a comprehensive prompt for content analysis based on prompt engineering best practices.
        
        Args:
            title: Title of the archive
            media_types: List of media types
            tags: List of tags
            description: User-provided description
            
        Returns:
            Enhanced comprehensive analysis prompt following best practices
        """
        media_types_str = ", ".join(media_types)
        tags_str = ", ".join(tags) if tags else "None provided"
        
        # Build media-specific instructions
        media_instructions = []
        if "image" in media_types:
            media_instructions.append(
                "**Images**: Analyze visual composition including objects, people, settings, text overlay, "
                "color schemes, lighting, perspective, mood, and any symbols or logos. Identify any readable text, "
                "brands, locations, or distinctive visual elements. Note the style (photography, illustration, diagram, etc.) "
                "and potential purpose or context."
            )
        if "video" in media_types:
            media_instructions.append(
                "**Videos**: Analyze visual and auditory elements including scenes, actions, dialogue, narration, "
                "background music, sound effects, pacing, transitions, camera movements, settings, participants, "
                "narrative structure, key moments, and overall message or story arc."
            )
        if "audio" in media_types:
            media_instructions.append(
                "**Audio**: Analyze spoken content including speakers, topics discussed, tone and emotion, "
                "background sounds, music (if any), key messages, questions and answers, important statements or quotes, "
                "contextual cues, and overall purpose or theme."
            )
        if "document" in media_types:
            media_instructions.append(
                "**Documents**: Extract key information including main topics, headings, bullet points, data tables, "
                "statistics, dates, names, organizations, conclusions, recommendations, and structural organization. "
                "Identify document type (report, article, contract, etc.) and summarize content systematically."
            )
        
        media_guidance = "\n".join(f"- {instr}" for instr in media_instructions) if media_instructions else "- Analyze all media types comprehensively"
        
        prompt = f"""# Role and Task Assignment

You are an expert content analyst and information extraction specialist with expertise in multimodal content analysis. Your task is to comprehensively analyze the uploaded materials and generate a detailed, structured summary that captures all essential information, context, and insights.

---

# Context Information

**Archive Title**: {title}
**Media Types Present**: {media_types_str}
**User Tags**: {tags_str}
**User Description**: {description if description else "Not provided by user"}

---

# Analysis Framework

Follow this structured approach to ensure comprehensive analysis:

## Step 1: Initial Assessment
Think step-by-step:
1. First, identify what types of content are present in the uploaded materials
2. Determine the primary purpose or intent of the content
3. Note any relationships or connections between different media files
4. Consider how the user-provided title, tags, and description relate to the actual content

## Step 2: Content-Specific Analysis

For each type of media present, perform detailed analysis:

{media_guidance}

## Step 3: Cross-Media Analysis (if multiple files)
- Identify relationships and connections between different files
- Note any recurring themes, concepts, or information across media
- Determine if files tell a cohesive story or relate to the same topic
- Highlight any contradictions or complementary information

## Step 4: Information Extraction

Extract and organize the following information systematically:

**Quantitative Data**:
- Numbers, statistics, measurements, percentages, dates, times
- Financial figures, quantities, metrics, scores, ratings

**Qualitative Information**:
- Names of people, places, organizations, products, brands
- Key concepts, themes, topics, subjects
- Emotions, opinions, perspectives expressed
- Events, actions, processes described

**Notable Elements**:
- Important quotes or statements (use exact wording when possible)
- Technical terms, jargon, or specialized vocabulary
- References to other sources, documents, or external information
- Calls to action, recommendations, or conclusions

## Step 5: Contextual and Semantic Analysis

- **Temporal Context**: Identify time periods, dates, chronological sequences, or temporal relationships
- **Spatial Context**: Note locations, geographical references, spatial relationships, or environments
- **Cultural/Social Context**: Identify cultural references, social norms, historical significance, or societal implications
- **Professional Context**: Recognize industry-specific content, professional domains, or technical contexts
- **Implicit Meanings**: Infer subtext, implied messages, underlying themes, or unstated connections
- **Significance**: Assess importance, relevance, urgency, or impact of the content

## Step 6: Synthesis and Summary Generation

Create a comprehensive summary that:

1. **Executive Overview** (2-3 sentences):
   - High-level summary of what the content is about
   - Primary purpose and main subject matter

2. **Detailed Content Summary** (structured by topic/theme):
   - Organized breakdown of key information
   - Each major topic or theme as a separate section
   - Include specific details, facts, and insights

3. **Key Findings and Insights**:
   - Most important information extracted
   - Notable patterns, trends, or observations
   - Significant facts, figures, or statements
   - Any surprising or noteworthy elements

4. **Contextual Relevance**:
   - How content relates to the provided title
   - Alignment with user tags and description
   - Additional context that may not have been captured in user metadata

---

# Output Requirements

**Structure**: Provide your analysis as a well-organized, readable text that flows naturally while maintaining clear structure.

**Depth**: Be thorough and detailed. Include specific information, facts, figures, names, dates, and other concrete details rather than vague generalizations.

**Accuracy**: Focus on factual information present in the content. Distinguish between what is explicitly stated versus what is inferred.

**Completeness**: Ensure all significant information from all uploaded files is captured. Do not omit important details.

**Clarity**: Use clear, professional language. Organize information logically and make it easy to understand.

**Length**: The summary should be comprehensive enough to capture all essential information (typically 500-1500 words depending on content volume and complexity).

---

# Instructions

1. Analyze ALL uploaded materials systematically
2. Consider the user-provided context (title, tags, description) but prioritize actual content analysis
3. Be thorough in extracting both explicit and implicit information
4. Maintain objectivity while noting any subjective elements (emotions, opinions) present
5. Organize information logically for easy comprehension
6. Ensure the summary is self-contained and understandable without viewing the original files

Begin your analysis now and provide the comprehensive summary."""
        
        return prompt.strip()
    
    async def upload_files_to_genai(
        self,
        files: List[UploadFile]
    ) -> tuple[List, List[str], List[str]]:
        """
        Upload files to Supabase storage and Google GenAI, returning file objects and metadata.
        
        Args:
            files: List of uploaded files
            
        Returns:
            Tuple containing:
                - List of uploaded file objects from Google GenAI
                - List of Supabase storage paths
                - List of Google GenAI file identifiers
            
        Raises:
            HTTPException: If file upload fails
        """
        if not files:
            raise HTTPException(status_code=400, detail="At least one file must be uploaded")

        uploaded_files: List = []
        storage_paths: List[str] = []
        genai_file_ids: List[str] = []
        temp_file_paths: List[str] = []
        
        try:
            for file in files:
                content = await file.read()
                mime_type = file.content_type or "application/octet-stream"
                storage_path = await self._upload_file_to_supabase_storage(
                    self._build_storage_path(file.filename),
                    content,
                    mime_type,
                )
                storage_paths.append(storage_path)

                suffix = Path(file.filename or "").suffix
                temp_path = self._create_temp_file(content, suffix)
                temp_file_paths.append(temp_path)

                try:
                    loop = asyncio.get_event_loop()

                    def upload_file():
                        return self.client.files.upload(
                            file=temp_path,
                            config=types.UploadFileConfig(
                                display_name=file.filename or "uploaded_file",
                                mime_type=mime_type,
                            ),
                        )

                    uploaded_file = await loop.run_in_executor(
                        self._executor,
                        upload_file,
                    )

                    max_wait_time = 300
                    wait_interval = 2
                    elapsed_time = 0

                    while uploaded_file.state == "PROCESSING" and elapsed_time < max_wait_time:
                        await asyncio.sleep(wait_interval)
                        elapsed_time += wait_interval

                        file_name = uploaded_file.name

                        def get_file():
                            return self.client.files.get(name=file_name)

                        uploaded_file = await loop.run_in_executor(
                            self._executor,
                            get_file,
                        )

                    if uploaded_file.state != "ACTIVE":
                        raise HTTPException(
                            status_code=500,
                            detail=f"File {file.filename} failed to process. State: {uploaded_file.state}",
                        )

                    uploaded_files.append(uploaded_file)
                    genai_file_ids.append(uploaded_file.name)

                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to upload file {file.filename}: {exc}",
                    ) from exc

            return uploaded_files, storage_paths, genai_file_ids
            
        finally:
            # Cleanup temporary files
            for temp_path in temp_file_paths:
                self._cleanup_temp_file(temp_path)
    
    async def analyze_content(
        self,
        uploaded_files: List,
        title: str,
        media_types: List[str],
        tags: List[str],
        description: str
    ) -> str:
        """
        Analyze uploaded content using Google GenAI and generate comprehensive summary.
        
        Args:
            uploaded_files: List of uploaded file objects from Google GenAI
            title: Title of the archive
            media_types: List of media types
            tags: List of tags
            description: User-provided description
            
        Returns:
            Comprehensive analysis summary as text
            
        Raises:
            HTTPException: If analysis fails
        """
        try:
            # Build content parts
            contents = []
            
            # Add the comprehensive prompt
            prompt = self._get_comprehensive_analysis_prompt(
                title=title,
                media_types=media_types,
                tags=tags,
                description=description
            )
            contents.append(prompt)
            
            # Add all uploaded files - file objects can be passed directly
            for uploaded_file in uploaded_files:
                contents.append(uploaded_file)
            
            # Generate content analysis (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            
            def generate_content():
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,  # Lower temperature for more focused, deterministic analysis
                        max_output_tokens=8192,  # Allow comprehensive responses
                        top_p=0.95,  # Nucleus sampling for diverse but focused responses
                        top_k=40,  # Limit vocabulary for more relevant outputs
                    )
                )
            
            response = await loop.run_in_executor(
                self._executor,
                generate_content
            )
            
            if not response.text:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate analysis. Empty response from model."
                )
            
            return response.text
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze content: {str(e)}"
            )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding using Google GenAI embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            HTTPException: If embedding generation fails
        """
        try:
            # Generate embedding (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            
            def embed_content():
                return self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                    )
                )
            
            response = await loop.run_in_executor(
                self._executor,
                embed_content
            )
            
            if not response.embeddings or len(response.embeddings) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate embedding. Empty response."
                )
            
            # Return the embedding values
            return response.embeddings[0].values
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {str(e)}"
            )
    
    async def _persist_archive_record(
        self,
        *,
        title: str,
        description: Optional[str],
        summary: str,
        embedding: List[float],
        media_types: List[str],
        tags: List[str],
        dates: List[datetime],
        storage_paths: List[str],
        genai_file_ids: List[str],
    ) -> dict:
        payload = {
            "title": title,
            "description": description or None,
            "summary": summary,
            "embedding": embedding,
            "media_types": media_types,
            "tags": tags if tags else [],
            "dates": [dt.isoformat() for dt in dates] if dates else [],
            "storage_paths": storage_paths,
            "genai_file_ids": genai_file_ids if genai_file_ids else [],
        }

        loop = asyncio.get_event_loop()

        def insert_record() -> dict:
            # In Supabase Python v2.x+, insert returns all fields by default
            response = (
                self.supabase_client.table("archives")
                .insert(payload)
                .execute()
            )
            data = response.data or []
            if not data:
                raise RuntimeError("Supabase insert returned no data")
            
            record = data[0]
            
            # Parse embedding if it's returned as a string (Supabase vector serialization)
            if "embedding" in record and isinstance(record["embedding"], str):
                try:
                    record["embedding"] = json.loads(record["embedding"])
                except (json.JSONDecodeError, TypeError):
                    # If it fails, keep the original value
                    pass
            
            return record

        try:
            return await loop.run_in_executor(self._executor, insert_record)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store archive metadata: {exc}",
            ) from exc
    
    async def process_archive(
        self,
        files: List[UploadFile],
        title: str,
        media_types: List[str],
        tags: List[str],
        description: str,
        dates: Optional[List[datetime]] = None,
    ) -> ArchiveResponse:
        """Complete archive processing pipeline."""

        uploaded_files, storage_paths, genai_file_ids = await self.upload_files_to_genai(files)
        file_uris = [file_obj.uri for file_obj in uploaded_files]

        summary = await self.analyze_content(
            uploaded_files=uploaded_files,
            title=title,
            media_types=media_types,
            tags=tags,
            description=description,
        )

        embedding = await self.generate_embedding(text=summary)

        archive_record = await self._persist_archive_record(
            title=title,
            description=description,
            summary=summary,
            embedding=embedding,
            media_types=media_types,
            tags=tags,
            dates=dates or [],
            storage_paths=storage_paths,
            genai_file_ids=genai_file_ids,
        )

        archive_record["file_uris"] = file_uris
        archive_record.setdefault("storage_paths", storage_paths)
        archive_record.setdefault("genai_file_ids", genai_file_ids)

        return ArchiveResponse(**archive_record)

