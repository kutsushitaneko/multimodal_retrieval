from __future__ import annotations

from PIL import Image

from app.react_agentic_rag import ReactAgenticRAGPipeline
from app.ui.workflow_agentic_events import (
    REFERENCE_TYPE_ALL,
    REFERENCE_TYPE_CAPTION_ONLY,
    WorkflowAgenticRAGEvents,
)


class ReactAgenticRAGEvents(WorkflowAgenticRAGEvents):
    """ReAct Agentic RAGタブ専用のイベントハンドラー。"""

    def register_react_agentic_rag_events(
        self,
        run_button,
        clear_button,
        question_input,
        uploaded_image,
        reference_type_radio,
        top_k_input,
        max_steps_input,
        answer_prompt_dropdown,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        controller_model,
        answer_text,
        referenced_images_gallery,
        trace_text,
        selection_reason_text,
    ):
        run_button.click(
            fn=self.run_react_agentic_rag,
            inputs=[
                question_input,
                uploaded_image,
                reference_type_radio,
                top_k_input,
                max_steps_input,
                answer_prompt_dropdown,
                vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
                controller_model,
            ],
            outputs=[answer_text, referenced_images_gallery, trace_text, selection_reason_text],
        )
        clear_button.click(
            fn=self.clear_react_agentic_rag,
            inputs=[],
            outputs=[
                question_input,
                uploaded_image,
                answer_text,
                referenced_images_gallery,
                trace_text,
                selection_reason_text,
            ],
            queue=False,
        )

    def run_react_agentic_rag(
        self,
        question,
        uploaded_image,
        reference_type,
        top_k,
        max_steps,
        answer_prompt_template,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        controller_model,
    ):
        def call_controller(prompt_text):
            return self._call_text_model(
                prompt_text,
                controller_model or vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
            )

        pipeline = ReactAgenticRAGPipeline(
            self.search_service,
            top_k=top_k,
            max_steps=max_steps,
            controller_llm_text_generator=call_controller,
        )

        def generate_answer(query, selected_evidence, documents):
            return self._generate_answer_with_vlm(
                query,
                selected_evidence,
                documents,
                reference_type,
                answer_prompt_template,
                vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
            )

        effective_reference_type = REFERENCE_TYPE_ALL if not (question or "").strip() and uploaded_image is not None else reference_type
        for result in pipeline.run_stream(
            question,
            uploaded_image=uploaded_image,
            answer_generator=generate_answer,
        ):
            yield self._format_react_agentic_rag_outputs(result, effective_reference_type)

    def _format_react_agentic_rag_outputs(self, result, reference_type):
        referenced_images = [
            evidence.image
            for evidence in result.selected_evidence
            if isinstance(evidence.image, Image.Image) and reference_type != REFERENCE_TYPE_CAPTION_ONLY
        ]
        gallery = self._create_referenced_images_gallery(referenced_images)
        return result.answer, gallery, result.trace, result.selection_reason

    def clear_react_agentic_rag(self):
        return self.clear_workflow_agentic_rag()
