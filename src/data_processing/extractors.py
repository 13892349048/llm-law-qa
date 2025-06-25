# multimodal data extractor 
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
from typing import Dict, Any, List
from pathlib import Path
import pymupdf4llm
import fitz
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
import pytesseract

logger = logging.getLogger(__name__)

class MultimodalExtractor:
    def __init__(self, 
                vision_model_name: str = "Salesforce/blip-image-captioning-base",
                ocr_language: str = "chi_sim+eng",
                device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 ):
        self.vision_model_name = vision_model_name
        self.ocr_language = ocr_language
        self.device = device

        try:
            self.vision_processor = BlipProcessor.from_pretrained(self.vision_model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(self.vision_model_name)
            self.vision_model.to(self.device)
            logger.info(f"Vision model loaded successfully from {self.vision_model_name}")
        except Exception as e:
            logger.error(f"Error loading vision model: {e}")
            self.vision_model = None
            self.vision_processor = None

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from a PDF file using OCR.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict[str, str]: A dictionary containing the extracted text.
        """

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting text from {pdf_path}")

        try:
            # use PyMuPDF4LLM to extract structured content from pdf 
            structed_content = pymupdf4llm.to_markdown(pdf_path)
            
            # get more content from pdf  low level api 
            doc = fitz.open(str(pdf_path)) 

            extracted_content = {
                "structured_content": structed_content,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "pages": [],
                "images": [],
                "tables": [],
                "metadata": self.__extract_metadata_from_pdf(doc)
            }
            
            # extract text from each page 
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_content = self._extract_content_from_page(page, page_num)
                extracted_content["pages"].append(page_content)
                extracted_content["images"].extend(page_content["images"])
                extracted_content["tables"].extend(page_content["tables"])

            doc.close()
            logger.info(f"Extracted {len(extracted_content['images'])} images and {len(extracted_content['tables'])} tables from {pdf_path}")
            return extracted_content
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise e
        
    def _extract_content_from_page(self, page, page_num: int) -> Dict[str, Any]:
        """
        Extract content from a page.

        Args:
            page: The page to extract content from.
            page_num: The number of the page.
        """
        page_content = {
            "page_number": page_num+1,
            "text": [],
            "images": [],
            "tables": [],
            "annotations": []
        }

        try:
            # extract text from page 
            text = page.get_text()
            page_content["text"].append(text)
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num + 1}: {e}")

        try:
            # extract images from page 
            images = self._extract_images_from_page(page, page_num)
            page_content["images"].extend(images)
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num + 1}: {e}")
        
        try:
            # extract tables from page 
            tables = self._extract_tables_from_page(page, page_num)
            page_content["tables"].extend(tables)
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num + 1}: {e}")
        
        return page_content
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract images from a page.

        Args:
            page: The page to extract images from.
            page_num: The number of the page.
        """

        images = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):   
            try:
                xref = img[0]
                # pix = fitz.Pixmap(page, xref)
                doc = page.parent
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                # transform to PIL
                pil_image = Image.open(BytesIO(image_bytes))

                #OCR get text from image 
                ocr_text = self._ocr_image(pil_image)

                #vision model get caption from image 
                description = self._generate_image_description(pil_image)

                #encode image to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()

                #get image rectangle from the page 
                image_rect = page.get_image_rects(xref)
                if image_rect:
                    rect = image_rect[0]
                    bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                else:
                    bbox = None

                #append to images list 
                image_info = {
                    "page_number": page_num + 1,
                    "image_base64": image_base64,
                    "ocr_text": ocr_text,
                    "description": description,
                    "image_index": img_index,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "bbox": bbox,
                    "image_ext": base_image.get("ext", "png"),  #img format
                    "image_size": len(image_bytes) #img size
                }
                images.append(image_info)

            except Exception as e:
                logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
        
        return images

    def _ocr_image(self, pil_image: Image.Image) -> str:
        """
        OCR the image.
        """
        try:
            #pretrained to improve ocr accuracy  
            processed_image = self._preprocess_image(pil_image)

            #ocr 
            ocr_text = pytesseract.image_to_string(processed_image, lang=self.ocr_language, config="--oem 3 --psm 6")

            return ocr_text.strip()
        
        except Exception as e:
            logger.error(f"Error OCRing image: {e}")
            return ""

    def _preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Preprocess the image.
        """
        try:
            # transform to numpy array 
            image_array = np.array(pil_image)

            #cv2 to improve ocr accuracy  grey image
            if len(image_array.shape) == 3:
                grey = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                grey = image_array

            #gaussian blur to reduce noise 
            blurred = cv2.GaussianBlur(grey, (5, 5), 0)

            #auto threshold to get binary image 
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            #convert to PIL image 
            pil_image = Image.fromarray(thresh)

            return pil_image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return pil_image

    def _generate_image_description(self, pil_image: Image.Image) -> str:
        """
        Get the description of the image.
        """
        if self.vision_model is None or self.vision_processor is None:
            logger.warning("Vision model not loaded, skipping image description generation")
            return ""
        
        try:
            # pre deal image 
            inputs = self.vision_processor(pil_image, return_tensors="pt").to(self.device)

            #generate description 
            with torch.no_grad():
                outputs = self.vision_model.generate(**inputs, max_new_tokens=50)
                description = self.vision_processor.decode(outputs[0], skip_special_tokens=True)
            
            return description
        
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return ""
    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables from a page.
        """
        tables = []
        
        try:
            page_tables = page.find_tables()

            for table_index, table in enumerate(page_tables):
                try:
                    #get data from table 
                    table_data = table.extract()

                    #get table rectangle 
                    bbox = table.bbox

                    #table transform to structured data 
                    structured_data = self._transform_table_to_structured_data(table_data)

                    table_info = {
                        "page_number": page_num + 1,
                        "table_index": table_index,
                        "bbox": bbox,
                        "raw_data": table_data,
                        "structured_data": structured_data,
                        "row_count": len(table_data) if table_data else 0,
                        "column_count": len(table_data[0]) if table_data and table_data[0] else 0
                    }
                    tables.append(table_info)
                except Exception as e:
                    logger.error(f"Error extracting table {table_index} from page {page_num + 1}: {e}")
            logger.info(f"Extracted {len(tables)} tables from page {page_num + 1}")
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num + 1}: {e}")
        return tables
    
    def _transform_table_to_structured_data(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """
        Transform table data to structured data.
        """
        if not table_data:
            return {}
        
        try:
            #get table header 
            headers = table_data[0] if table_data else []
            #get table body 
            body = table_data[1:] if table_data else []

            # clear empty rows and columns 
            headers = [header.strip() if header else f"column_{i+1}" for i, header in enumerate(headers)]

            #dict 
            structured_data = []

            for row in body:
                row_dict = {}
                for i, cell in enumerate(row):
                    if i < len(headers):
                        row_dict[headers[i]] = cell.strip()
                    else:
                        row_dict[f"column_{i+1}"] = cell.strip()
                structured_data.append(row_dict)

            return {
                "headers": headers,
                "rows": structured_data,
                "raw_data": table_data
            }
        
        except Exception as e:
            logger.error(f"Error transforming table to structured data: {e}")
            return {}
    
    def __extract_metadata_from_pdf(self, doc) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        """

        try:
            metadata = doc.metadata
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creation_date", ""),
                "modification_date": metadata.get("modification_date", ""),
                "page_count": doc.page_count,
            }
        
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {e}")
            return {}

mu = MultimodalExtractor()

pdf_path = "/home/deepuser/Downloads/test1.pdf"

extracted_content = mu.extract_from_pdf(pdf_path)

print("extracted_content: \n", extracted_content["metadata"])
        