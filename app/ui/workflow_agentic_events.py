import os
import tempfile

import gradio as gr
from PIL import Image

from app.agentic_rag_common import REFERENCED_GALLERY_ELEM_CLASS
from app.workflow_agentic_rag import WorkflowAgenticRAGPipeline
from app.nlp_service import NLPService
from app.vlm_service_factory import VLMServiceFactory


REFERENCE_TYPE_ALL = "すべて"
REFERENCE_TYPE_CAPTION_ONLY = "キャプションのみ"
REFERENCE_TYPE_IMAGE_ONLY = "画像のみ"


class WorkflowAgenticRAGEvents:
    """Workflow Agentic RAGタブ専用のイベントハンドラー。"""

    RAG_TYPE_LABEL = "Workflow Agentic RAG"

    def __init__(self, search_service):
        self.search_service = search_service
        self.answer_vlm_service = VLMServiceFactory.create_answer_vlm_service()

    def register_workflow_agentic_rag_events(
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
        decompose_temperature,
        decompose_max_tokens,
        decompose_oci_region,
        sufficiency_model,
        sufficiency_temperature,
        sufficiency_max_tokens,
        sufficiency_oci_region,
        followup_query_model,
        followup_query_temperature,
        followup_query_max_tokens,
        followup_query_oci_region,
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
            fn=self.run_workflow_agentic_rag,
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
                decompose_temperature,
                decompose_max_tokens,
                decompose_oci_region,
                sufficiency_model,
                sufficiency_temperature,
                sufficiency_max_tokens,
                sufficiency_oci_region,
                followup_query_model,
                followup_query_temperature,
                followup_query_max_tokens,
                followup_query_oci_region,
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
            fn=self.clear_workflow_agentic_rag,
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

    def register_vlm_settings_events(
        self,
        vlm_service_provider,
        vlm_model,
        vlm_temperature,
        vlm_max_tokens,
        vlm_oci_region,
        decompose_model=None,
        decompose_temperature=None,
        decompose_max_tokens=None,
        decompose_oci_region=None,
        sufficiency_model=None,
        sufficiency_temperature=None,
        sufficiency_max_tokens=None,
        sufficiency_oci_region=None,
        followup_query_model=None,
        followup_query_temperature=None,
        followup_query_max_tokens=None,
        followup_query_oci_region=None,
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
            fn=lambda temp: self.answer_vlm_service.update_current_vlm_settings(temperature=temp),
            inputs=[vlm_temperature],
            outputs=[],
        )
        vlm_max_tokens.change(
            fn=lambda tokens: self.answer_vlm_service.update_current_vlm_settings(max_tokens=tokens),
            inputs=[vlm_max_tokens],
            outputs=[],
        )
        vlm_oci_region.change(
            fn=lambda region: self.answer_vlm_service.update_current_vlm_settings(
                oci_region=self.answer_vlm_service.resolve_oci_region_id(region)
            ),
            inputs=[vlm_oci_region],
            outputs=[],
        )
        for model, temperature, max_tokens, oci_region in [
            (decompose_model, decompose_temperature, decompose_max_tokens, decompose_oci_region),
            (sufficiency_model, sufficiency_temperature, sufficiency_max_tokens, sufficiency_oci_region),
            (followup_query_model, followup_query_temperature, followup_query_max_tokens, followup_query_oci_region),
        ]:
            if model is None or temperature is None or max_tokens is None or oci_region is None:
                continue
            model.change(
                fn=self.agentic_model_changed,
                inputs=[model],
                outputs=[temperature, max_tokens, oci_region],
                queue=False,
            )

    def vlm_service_provider_changed(self, service_provider):
        return self.answer_vlm_service.service_provider_changed(service_provider)

    def vlm_model_changed(self, model):
        self.answer_vlm_service.update_current_vlm_settings(model=model)
        return self.answer_vlm_service.model_changed(model)

    def agentic_model_changed(self, model):
        return self.answer_vlm_service.create_model_parameter_components(model)

    def run_workflow_agentic_rag(
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
        decompose_temperature,
        decompose_max_tokens,
        decompose_oci_region,
        sufficiency_model,
        sufficiency_temperature,
        sufficiency_max_tokens,
        sufficiency_oci_region,
        followup_query_model,
        followup_query_temperature,
        followup_query_max_tokens,
        followup_query_oci_region,
    ):
        def call_workflow_selection_model(prompt_text):
            return self._call_text_vlm(
                prompt_text,
                vlm_model,
                vlm_temperature,
                vlm_max_tokens,
                vlm_oci_region,
                uploaded_image=uploaded_image,
            )

        def call_agentic_step_model(prompt_text, model, temperature, max_tokens, oci_region):
            return self._call_text_model(
                prompt_text,
                model or vlm_model,
                temperature,
                max_tokens,
                oci_region,
            )

        pipeline = WorkflowAgenticRAGPipeline(
            self.search_service,
            top_k=top_k,
            max_iterations=max_iterations,
            llm_text_generator=call_workflow_selection_model,
            decompose_llm_text_generator=lambda prompt: call_agentic_step_model(
                prompt,
                decompose_model,
                decompose_temperature,
                decompose_max_tokens,
                decompose_oci_region,
            ),
            sufficiency_llm_text_generator=lambda prompt: call_agentic_step_model(
                prompt,
                sufficiency_model,
                sufficiency_temperature,
                sufficiency_max_tokens,
                sufficiency_oci_region,
            ),
            followup_llm_text_generator=lambda prompt: call_agentic_step_model(
                prompt,
                followup_query_model,
                followup_query_temperature,
                followup_query_max_tokens,
                followup_query_oci_region,
            ),
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
            yield self._format_workflow_agentic_rag_outputs(result, effective_reference_type)

    def _format_workflow_agentic_rag_outputs(self, result, reference_type):
        referenced_images = []
        referenced_details = []
        for evidence in result.selected_evidence:
            if isinstance(evidence.image, Image.Image) and reference_type != REFERENCE_TYPE_CAPTION_ONLY:
                referenced_images.append(evidence.image)
                referenced_details.append(self._build_referenced_detail(evidence))
        gallery = self._create_referenced_images_gallery(referenced_images)
        return result.answer, gallery, result.trace, result.selection_reason, referenced_details

    @staticmethod
    def _build_referenced_detail(evidence):
        return {
            "file_name": evidence.file_name or "",
            "image_id": "" if evidence.image_id is None else str(evidence.image_id),
            "caption": evidence.caption or "",
            "distance": evidence.distance,
        }

    def _register_referenced_detail_events(
        self,
        referenced_images_gallery,
        referenced_details_state,
        detail_filename_text,
        detail_image_id_text,
        detail_similarity_text,
        detail_caption_text,
    ):
        if (
            referenced_images_gallery is None
            or referenced_details_state is None
            or detail_filename_text is None
            or detail_image_id_text is None
            or detail_similarity_text is None
            or detail_caption_text is None
        ):
            return
        referenced_images_gallery.select(
            fn=self._handle_referenced_selection,
            inputs=[referenced_details_state],
            outputs=[detail_filename_text, detail_image_id_text, detail_similarity_text, detail_caption_text],
        )

    @staticmethod
    def _handle_referenced_selection(evt: gr.SelectData, details):
        items = details or []
        if evt.index is None or evt.index < 0 or evt.index >= len(items):
            return "", "", "", ""
        item = items[evt.index]
        distance = item.get("distance")
        similarity_text = "" if distance is None else f"{-1 * distance:.4f}"
        return (
            item.get("file_name", ""),
            item.get("image_id", ""),
            similarity_text,
            item.get("caption", ""),
        )

    def clear_workflow_agentic_rag(self):
        return (
            "",
            None,
            "",
            self._create_referenced_images_gallery([]),
            "",
            "",
            "",
            "",
            "",
            "",
            [],
        )

    def _create_referenced_images_gallery(self, referenced_images):
        return gr.Gallery(
            label="参照した画像",
            value=referenced_images or [],
            show_label=True,
            columns=[4],
            rows=[2],
            object_fit="contain",
            container=True,
            preview=False,
            allow_preview=True,
            elem_classes=[REFERENCED_GALLERY_ELEM_CLASS],
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
        uploaded_image=None,
    ):
        has_uploaded_image = isinstance(uploaded_image, Image.Image)
        if not selected_evidence and not has_uploaded_image:
            return "❌ 回答に使用できる検索結果が見つかりませんでした。"

        prompt = self._load_answer_prompt(answer_prompt_template)
        final_prompt = prompt.replace("{query_text}", query or "").replace("{documents}", documents or "")
        if "{query_text}" not in prompt and "{documents}" not in prompt:
            final_prompt = f"{prompt}\n\n質問:\n{query}\n\n参照情報:\n{documents}"

        if has_uploaded_image:
            final_prompt = (
                "【画像の並び順】1枚目はユーザーがアップロードした画像です。"
                "検索で見つかった参照画像がある場合は2枚目以降です。\n\n"
                f"{final_prompt}"
            )

        image_paths = []
        try:
            if reference_type != REFERENCE_TYPE_CAPTION_ONLY:
                image_paths = self._save_evidence_images(selected_evidence)
            if has_uploaded_image:
                image_paths.insert(0, self._save_uploaded_image(uploaded_image))

            nlp_service = NLPService(self.answer_vlm_service)
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
            reference_name_items = []
            if has_uploaded_image:
                reference_name_items.append("ユーザーがアップロードした画像")
            reference_name_items.extend(evidence.file_name for evidence in selected_evidence if evidence.file_name)
            reference_names = "」「".join(reference_name_items) or "参照情報"
            return f"（{self.RAG_TYPE_LABEL} が「{reference_names}」を参照して回答しました）\n\n{answer}"
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
            nlp_service = NLPService(self.answer_vlm_service)
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
        nlp_service = NLPService(self.answer_vlm_service)
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
            print(f"Workflow Agentic RAG回答生成プロンプト読み込みエラー: {exc}")
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

    def _save_uploaded_image(self, uploaded_image):
        image = uploaded_image
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            return temp_file.name

    def _create_blank_image(self):
        blank_image = Image.new("RGB", (32, 32), color="white")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as temp_file:
            blank_image.save(temp_file, format="PNG")
            return temp_file.name
