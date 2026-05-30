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

    RAG_TYPE_LABEL = "ReAct Agentic RAG"

    def register_vlm_settings_events(
        self,
        vlm_service_provider,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        controller_model=None,
        controller_temperature=None,
        controller_max_tokens=None,
        controller_oci_region=None,
    ):
        super().register_vlm_settings_events(
            vlm_service_provider,
            vlm_model,
            vlm_temperature,
            vlm_max_tokens,
            vlm_oci_region,
        )
        if (
            controller_model is None
            or controller_temperature is None
            or controller_max_tokens is None
            or controller_oci_region is None
        ):
            return
        controller_model.change(
            fn=self.agentic_model_changed,
            inputs=[controller_model],
            outputs=[controller_temperature, controller_max_tokens, controller_oci_region],
            queue=False,
        )

    def register_react_agentic_rag_events(
        self,
        run_button,
        clear_button,
        question_input,
        uploaded_image,
        reference_type_radio,
        top_k_input,
        max_steps_input,
        max_selected_evidence_input,
        answer_prompt_dropdown,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        controller_model,
        controller_temperature,
        controller_max_tokens,
        controller_oci_region,
        answer_text,
        referenced_images_gallery,
        trace_text,
        selection_reason_text,
        detail_filename_text=None,
        detail_image_id_text=None,
        detail_similarity_text=None,
        detail_caption_text=None,
        referenced_details_state=None,
    ):
        run_button.click(
            fn=self.run_react_agentic_rag,
            inputs=[
                question_input,
                uploaded_image,
                top_k_input,
                max_steps_input,
                reference_type_radio,
                max_selected_evidence_input,
                answer_prompt_dropdown,
                vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
                controller_model,
                controller_temperature,
                controller_max_tokens,
                controller_oci_region,
            ],
            outputs=[answer_text, referenced_images_gallery, trace_text, selection_reason_text, referenced_details_state],
        )
        self._register_referenced_detail_events(
            referenced_images_gallery,
            referenced_details_state,
            detail_filename_text,
            detail_image_id_text,
            detail_similarity_text,
            detail_caption_text,
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
                detail_filename_text,
                detail_image_id_text,
                detail_similarity_text,
                detail_caption_text,
                referenced_details_state,
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
        max_selected_evidence,
        answer_prompt_template,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        controller_model,
        controller_temperature,
        controller_max_tokens,
        controller_oci_region,
    ):
        effective_controller_model = str(controller_model or "").strip()

        def call_controller(prompt_text):
            return self._call_text_model(
                prompt_text,
                effective_controller_model,
                controller_temperature,
                controller_max_tokens,
                controller_oci_region,
            )

        pipeline = ReactAgenticRAGPipeline(
            self.search_service,
            top_k=top_k,
            max_steps=max_steps,
            max_selected_evidence=max_selected_evidence,
            controller_llm_text_generator=call_controller if effective_controller_model else None,
            controller_model_name=effective_controller_model,
            finalize_verifier_llm_text_generator=call_controller if effective_controller_model else None,
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
                uploaded_image=uploaded_image,
            )

        effective_reference_type = REFERENCE_TYPE_ALL if not (question or "").strip() and uploaded_image is not None else reference_type
        for result in pipeline.run_stream(
            question,
            uploaded_image=uploaded_image,
            answer_generator=generate_answer,
        ):
            yield self._format_react_agentic_rag_outputs(result, effective_reference_type)

    def _format_react_agentic_rag_outputs(self, result, reference_type):
        referenced_images = []
        referenced_details = []
        for evidence in result.selected_evidence:
            if isinstance(evidence.image, Image.Image) and reference_type != REFERENCE_TYPE_CAPTION_ONLY:
                referenced_images.append(evidence.image)
                referenced_details.append(self._build_referenced_detail(evidence))
        gallery = self._create_referenced_images_gallery(referenced_images)
        return result.answer, gallery, result.trace, result.selection_reason, referenced_details

    def clear_react_agentic_rag(self):
        return self.clear_workflow_agentic_rag()

    def _resolve_controller_generation_settings(self, controller_model):
        if not controller_model:
            return 0.0, 4096, None
        temperature = self.answer_vlm_service.get_model_default_temperature(controller_model)
        max_tokens = self.answer_vlm_service.get_model_default_tokens(controller_model)
        default_region = self.answer_vlm_service.get_model_default_region(controller_model)
        return temperature, max_tokens, default_region
