import os
import tempfile

import gradio as gr
from PIL import Image

from app.agentic_rag import AgenticRAGPipeline
from app.nlp_service import NLPService
from app.vlm_service_factory import VLMServiceFactory


REFERENCE_TYPE_ALL = "すべて"
REFERENCE_TYPE_CAPTION_ONLY = "キャプションのみ"
REFERENCE_TYPE_IMAGE_ONLY = "画像のみ"


class AgenticRAGEvents:
    """Agentic RAGタブ専用のイベントハンドラー。"""

    def __init__(self, search_service):
        self.search_service = search_service
        self.agentic_vlm_service = VLMServiceFactory.create_answer_vlm_service()

    def register_agentic_rag_events(
        self,
        run_button,
        clear_button,
        question_input,
        uploaded_image,
        reference_type_radio,
        top_k_input,
        max_iterations_input,
        answer_prompt_dropdown,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        decompose_model,
        sufficiency_model,
        followup_query_model,
        answer_text,
        referenced_images_gallery,
        trace_text,
        selection_reason_text,
    ):
        run_button.click(
            fn=self.run_agentic_rag,
            inputs=[
                question_input,
                uploaded_image,
                reference_type_radio,
                top_k_input,
                max_iterations_input,
                answer_prompt_dropdown,
                vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
                decompose_model,
                sufficiency_model,
                followup_query_model,
            ],
            outputs=[answer_text, referenced_images_gallery, trace_text, selection_reason_text],
        )
        clear_button.click(
            fn=self.clear_agentic_rag,
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

    def register_vlm_settings_events(
        self,
        vlm_service_provider,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
    ):
        vlm_service_provider.change(
            fn=self.vlm_service_provider_changed,
            inputs=[vlm_service_provider],
            outputs=[vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region],
            queue=False,
        )
        vlm_model.change(
            fn=self.vlm_model_changed,
            inputs=[vlm_model],
            outputs=[vlm_temperature, vlm_max_tokens, vlm_oci_region],
            queue=False,
        )
        vlm_temperature.change(
            fn=lambda temp: self.agentic_vlm_service.update_current_vlm_settings(temperature=temp),
            inputs=[vlm_temperature],
            outputs=[],
        )
        vlm_max_tokens.change(
            fn=lambda tokens: self.agentic_vlm_service.update_current_vlm_settings(max_tokens=tokens),
            inputs=[vlm_max_tokens],
            outputs=[],
        )
        vlm_oci_region.change(
            fn=lambda region: self.agentic_vlm_service.update_current_vlm_settings(
                oci_region=self.agentic_vlm_service.resolve_oci_region_id(region)
            ),
            inputs=[vlm_oci_region],
            outputs=[],
        )

    def vlm_service_provider_changed(self, service_provider):
        return self.agentic_vlm_service.service_provider_changed(service_provider)

    def vlm_model_changed(self, model):
        self.agentic_vlm_service.update_current_vlm_settings(model=model)
        return self.agentic_vlm_service.model_changed(model)

    def run_agentic_rag(
        self,
        question,
        uploaded_image,
        reference_type,
        top_k,
        max_iterations,
        answer_prompt_template,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        decompose_model,
        sufficiency_model,
        followup_query_model,
    ):
        def call_agentic_judge(prompt_text):
            return self._call_text_vlm(
                prompt_text,
                vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
            )

        def call_agentic_step_model(prompt_text, model):
            return self._call_text_model(
                prompt_text,
                model or vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
            )

        pipeline = AgenticRAGPipeline(
            self.search_service,
            top_k=top_k,
            max_iterations=max_iterations,
            llm_text_generator=call_agentic_judge,
            decompose_llm_text_generator=lambda prompt: call_agentic_step_model(prompt, decompose_model),
            sufficiency_llm_text_generator=lambda prompt: call_agentic_step_model(prompt, sufficiency_model),
            followup_llm_text_generator=lambda prompt: call_agentic_step_model(prompt, followup_query_model),
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

        result = pipeline.run(
            question,
            uploaded_image=uploaded_image,
            answer_generator=generate_answer,
        )
        referenced_images = [
            evidence.image
            for evidence in result.selected_evidence
            if isinstance(evidence.image, Image.Image) and reference_type != REFERENCE_TYPE_CAPTION_ONLY
        ]
        gallery = self._create_referenced_images_gallery(referenced_images)
        trace = (
            f"{result.trace}\n"
            f"- 参照画像ギャラリー: selected evidence {len(result.selected_evidence)} 件, "
            f"画像表示 {len(referenced_images)} 件"
        )
        return result.answer, gallery, trace, result.selection_reason

    def clear_agentic_rag(self):
        return (
            "",
            None,
            "",
            self._create_referenced_images_gallery([]),
            "",
            "",
        )

    def _create_referenced_images_gallery(self, referenced_images):
        image_count = len(referenced_images or [])
        rows = max(1, min(3, (image_count + 3) // 4))
        return gr.Gallery(
            label="参照した画像",
            value=referenced_images or [],
            columns=4,
            rows=rows,
            height=240 * rows,
            object_fit="contain",
            visible=bool(referenced_images),
        )

    def _generate_answer_with_vlm(
        self,
        query,
        selected_evidence,
        documents,
        reference_type,
        answer_prompt_template,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
    ):
        if not selected_evidence:
            return "❌ 回答に使用できる検索結果が見つかりませんでした。"

        prompt = self._load_answer_prompt(answer_prompt_template)
        final_prompt = prompt.replace("{query_text}", query or "").replace("{documents}", documents or "")
        if "{query_text}" not in prompt and "{documents}" not in prompt:
            final_prompt = f"{prompt}\n\n質問:\n{query}\n\n参照情報:\n{documents}"

        image_paths = []
        try:
            if reference_type != REFERENCE_TYPE_CAPTION_ONLY:
                image_paths = self._save_evidence_images(selected_evidence)

            nlp_service = NLPService(self.agentic_vlm_service)
            if image_paths:
                answer = nlp_service.generate_answer_with_vlm_images(
                    image_paths=image_paths,
                    vlm_model=vlm_model,
                    prompt_text=final_prompt,
                    temperature=vlm_temperature,
                    max_tokens=vlm_max_tokens,
                    oci_region=vlm_oci_region,
                )
            else:
                blank_image_path = self._create_blank_image()
                image_paths.append(blank_image_path)
                answer = nlp_service.generate_caption_with_vlm(
                    image_path=blank_image_path,
                    vlm_model=vlm_model,
                    prompt_text=final_prompt,
                    temperature=vlm_temperature,
                    max_tokens=vlm_max_tokens,
                    oci_region=vlm_oci_region,
                )

            if not answer:
                return "❌ 回答の生成に失敗しました。"
            reference_names = "」「".join(evidence.file_name for evidence in selected_evidence)
            return f"（Agentic RAG が「{reference_names}」を参照して回答しました）\n\n{answer}"
        finally:
            for image_path in image_paths:
                if image_path and os.path.exists(image_path):
                    os.unlink(image_path)

    def _call_text_vlm(
        self,
        prompt_text,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
    ):
        image_path = self._create_blank_image()
        try:
            nlp_service = NLPService(self.agentic_vlm_service)
            return nlp_service.generate_caption_with_vlm(
                image_path=image_path,
                vlm_model=vlm_model,
                prompt_text=prompt_text,
                temperature=vlm_temperature,
                max_tokens=vlm_max_tokens,
                oci_region=vlm_oci_region,
            )
        finally:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)

    def _call_text_model(
        self,
        prompt_text,
        model,
        temperature,
        max_tokens,
        oci_region,
    ):
        nlp_service = NLPService(self.agentic_vlm_service)
        return nlp_service.generate_text(
            model=model,
            prompt_text=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
            oci_region=oci_region,
        )

    def _load_answer_prompt(self, template_name):
        template_name = template_name or "デフォルト（回答生成）"
        prompt_path = os.path.join("answer_prompt", f"{template_name}.txt")
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                    return prompt_file.read()
        except Exception as exc:
            print(f"Agentic RAG回答生成プロンプト読み込みエラー: {exc}")
        return "質問に対して、参照情報に基づいて日本語で回答してください。\n\n質問: {query_text}\n\n参照情報:\n{documents}"

    def _save_evidence_images(self, selected_evidence):
        image_paths = []
        for evidence in selected_evidence:
            if not isinstance(evidence.image, Image.Image):
                continue
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as temp_file:
                evidence.image.save(temp_file, format="PNG")
                image_paths.append(temp_file.name)
        return image_paths

    def _create_blank_image(self):
        blank_image = Image.new("RGB", (32, 32), color="white")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as temp_file:
            blank_image.save(temp_file, format="PNG")
            return temp_file.name
