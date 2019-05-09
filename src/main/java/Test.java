import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.FloatBuffer;


public class Test {

    static final int WIDTH = 128;
    static final int SCAlE_WIDTH = 512;
    static final float speed = .05f;

    float[][] input = new float[1][100];
    float[] direction = new float[100];

    public static void main(String[] args) throws IOException, InterruptedException {
        new Test();
    }

    public Test() throws IOException, InterruptedException {

        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        ImageIcon imageIcon = new ImageIcon(new BufferedImage(SCAlE_WIDTH, SCAlE_WIDTH, BufferedImage.TYPE_INT_RGB));
        JLabel label = new JLabel(imageIcon);
        label.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                resetImage();
            }

            @Override
            public void mousePressed(MouseEvent e) {

            }

            @Override
            public void mouseReleased(MouseEvent e) {

            }

            @Override
            public void mouseEntered(MouseEvent e) {

            }

            @Override
            public void mouseExited(MouseEvent e) {

            }
        });
        frame.getContentPane().add(label);
//        frame.setUndecorated(true);
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        // load the model Bundle

        try (SavedModelBundle smb = SavedModelBundle.load("./model", "serve")) {
            Session sess = smb.session();
            Graph graph = smb.graph();
            resetImage();
            int cnt = 0;
            while (true) {
                cnt++;
                if (cnt % 100 == 0) {
                    resetImage();
                }
                for (int i = 0; i < 100; i++) {
                    if (input[0][i] < -1f || input[0][i] > 1f) {
                        direction[i] = -direction[i];
                    }
                    input[0][i] += direction[i];
                    direction[i] += ((float) (Math.random() * 2.0f) - 1f) / 500f;
                }
                normalize(direction, speed);
                Tensor t = Tensor.create(input);
                Tensor out = runAI(sess, t, "input_z", "generator/out");

//                Tensor discr = runAI(sess, out, "inputs_real", "discriminator/out");
//                FloatBuffer discrData = FloatBuffer.allocate(1);
//                discr.writeTo(discrData);
//                float discAr = discrData.array()[0];
//                System.out.println(discAr);

                BufferedImage image = scale(convert(out), SCAlE_WIDTH);
                imageIcon.setImage(image);
                label.repaint();
            }

        }
    }

    private void resetImage() {
        for (int i = 0; i < 100; i++) {
            input[0][i] = (float) (Math.random() * 2.0f - 1f);
            direction[i] = (float) ((Math.random() * 2.0f - 1f));
        }
        normalize(direction, speed);
    }


    /**
     * Generator input [100] output [1][128][128][3]
     *
     * @param sess
     * @param inputTensor
     * @return
     * @throws IOException
     * @throws FileNotFoundException
     */
    public static Tensor runAI(Session sess, Tensor inputTensor, String input, String output) throws IOException, FileNotFoundException {
        Tensor result = sess.runner()
                .feed(input, inputTensor)
                .fetch(output)
                .run().get(0);
        return result;
    }


    public BufferedImage convert(Tensor out) {
        FloatBuffer imageData = FloatBuffer.allocate(128 * 128 * 3);
        out.writeTo(imageData);
        imageData.rewind();
//        float min = Float.MAX_VALUE;
//        float max = Float.MIN_VALUE;
        float min = -1;
        float max = 1;
        /*while (imageData.hasRemaining()) {
            float f = imageData.get();
            if (f > max) max = f;
            if (f < min) min = f;
        }
        imageData.rewind();*/

        // fill rgbArray for BufferedImage
        int[] rgbArray = new int[WIDTH * WIDTH];
        float val = 0;
        int valI = 0;
        for (int y = WIDTH - 1; y > -1; --y) {
            for (int x = 0; x < WIDTH; ++x) {
                val = imageData.get();
                valI = (int) (((val - min) * 255) / (max - min));
                int r = valI << 16;
                val = imageData.get();
                valI = (int) (((val - min) * 255) / (max - min));
                int g = valI << 8;
                val = imageData.get();
                valI = (int) (((val - min) * 255) / (max - min));
                int b = valI;
                int i = ((WIDTH - 1) - y) * WIDTH + x;
                rgbArray[i] = r + g + b;
            }
        }

        // create and save image
        BufferedImage image = new BufferedImage(
                WIDTH, WIDTH, BufferedImage.TYPE_INT_RGB
        );
        image.setRGB(0, 0, WIDTH, WIDTH, rgbArray, 0, WIDTH);
        return image;
    }

    public BufferedImage scale(BufferedImage original, int size) {
        BufferedImage resized = new BufferedImage(size, size, original.getType());
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, size, size, 0, 0, original.getWidth(),
                original.getHeight(), null);
        g.dispose();
        return resized;
    }


    public void normalize(float[] input, float scale) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (float f : input) {
            if (f < min) min = f;
            if (f > max) max = f;
        }

        for (int i = 0; i < input.length; i++) {
            input[i] = (((input[i] - min) / (max - min)) * 2f - 1f) * scale;
        }
    }
}
