import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
CAMINHO_IMAGEM = BASE / "qr code (1).webp"
SAIDA_PNG = BASE / "qrcode_corrigido.png"


def carregar_imagem(caminho: Path):
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
    except ImportError:
        print("Instale o Pillow se o OpenCV não abrir a imagem: pip install pillow")
        sys.exit(1)
    pil = Image.open(caminho).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def decodificar(img) -> str:
    d, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return d or ""


def angulo_em_multiplos_de_90(img) -> int | None:
    """Qual rotação da original (0/90/180/270) já permite ler o QR."""
    for graus, flag in (
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ):
        r = img if flag is None else cv2.rotate(img, flag)
        if decodificar(r):
            return graus
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualizar",
        action="store_true",
        help="Mostra original e corrigida em janelas.",
    )
    args = parser.parse_args()

    if not CAMINHO_IMAGEM.is_file():
        print("Erro: arquivo não encontrado:", CAMINHO_IMAGEM)
        sys.exit(1)

    img = carregar_imagem(CAMINHO_IMAGEM)
    achado = angulo_em_multiplos_de_90(img)
    if achado is None:
        print("Não achei orientação legível em múltiplos de 90°.")
        sys.exit(1)

    # OpenCV usa ângulo positivo = anti-horário; alinhamos ao que funcionou no teste.
    angulo_correcao = -float(achado)

    altura, largura = img.shape[:2]
    centro = (largura // 2, altura // 2)
    borda = (255, 255, 255)

    M_rot = cv2.getRotationMatrix2D(centro, angulo_correcao, 1.0)
    rotacionada = cv2.warpAffine(
        img,
        M_rot,
        (largura, altura),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=borda,
    )

    # Translação em px (exemplo explícito; pode ajustar ou zerar se cortar o QR).
    dx, dy = 50, 30
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    corrigida = cv2.warpAffine(
        rotacionada,
        M_trans,
        (largura, altura),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=borda,
    )

    texto = decodificar(corrigida)
    cv2.imwrite(str(SAIDA_PNG), corrigida)

    print("Rotação de referência na original (múltiplo de 90°):", achado, "°")
    print("Ângulo aplicado em getRotationMatrix2D:", angulo_correcao, "°")
    print("Translação (dx, dy):", dx, dy)
    print("Imagem salva:", SAIDA_PNG)
    print("Conteúdo do QR:", texto)

    if args.visualizar:
        cv2.imshow("Original", img)
        cv2.imshow("Corrigido", corrigida)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
