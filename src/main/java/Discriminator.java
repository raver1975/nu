import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

import static org.bytedeco.javacpp.avutil.AV_PIX_FMT_ARGB;


public class Discriminator {

//    private static final String imageUrl = "https://imagesvc.timeincapp.com/v3/mm/image?url=https%3A%2F%2Fcdn-s3.si.com%2Fimages%2FX161321_TK3_13975-rawfinalWMWMweb1920.jpg&w=1000&q=70";
//    private static final String imageUrl = "https://i.ytimg.com/vi/ERRvSgL_dfA/maxresdefault.jpg";
//    private static final String imageUrl = "https://terezowens.com/wp-content/uploads/2017/04/kelly_rohrbach_2016_photo_sports_illustrated_x159794_tk3_09573_rawwmfinal1920-579x868.jpg";
//    private static final String imageUrl = "https://i.dailymail.co.uk/i/pix/2018/09/30/18/50D9F63000000578-6224641-A_10_Sports_Illustrated_model_Alexis_Ren_has_been_working_hard_t-m-87_1538328364533.jpg";

    private static final String imageUrl = "https://img.purch.com/w/660/aHR0cDovL3d3dy5saXZlc2NpZW5jZS5jb20vaW1hZ2VzL2kvMDAwLzA4OC85MTEvb3JpZ2luYWwvZ29sZGVuLXJldHJpZXZlci1wdXBweS5qcGVn";
//    private static final String imageUrl = "https://i.ytimg.com/vi/ZYifkcmIb-4/maxresdefault.jpg";

    private static final int WIDTH = 128;
    private int num_images = 1;
    private float[][][][] input = new float[num_images][128][128][3];

    public static void main(String[] args) throws IOException, InterruptedException {
        new Discriminator();
    }

    private Discriminator() throws IOException, InterruptedException {

        // load the model Bundle

        try (SavedModelBundle smb = SavedModelBundle.load("./model", "train")) {
            Session sess = smb.session();
            Graph graph = smb.graph();

            BufferedImage image= scale(new BufferedImage[]{ImageIO.read(new URL(imageUrl))},WIDTH)[0];

            int rgb;
            float r,g,b;
            for (int y = WIDTH - 1; y > -1; --y) {
                for (int x = 0; x < WIDTH; ++x) {
                    rgb = image.getRGB(x,y);
                    r = (rgb >> 16) & 0xFF;
                    g= (rgb >> 8) & 0xFF;
                    b = rgb & 0xFF;
                    input[0][y][x][0]=r/128f-1f;
                    input[0][y][x][1]=g/128f-1f;
                    input[0][y][x][2]=b/128f-1f;
                }
            }
            Tensor out = Tensor.create(input);
            Tensor discr = runAI(sess, out, "inputs_real", "discriminator/out");
            FloatBuffer discrData = FloatBuffer.allocate(1);
            discr.writeTo(discrData);
            discrData.rewind();
            float discAr = discrData.get();
            System.out.println("-------------------------");
            System.out.printf("%.16f", discAr);
            System.out.println("\n-------------------------");

            JFrame frame = new JFrame();
            frame.getContentPane().setLayout(new FlowLayout());
            ImageIcon imageIcon = new ImageIcon(convert(out)[0]);
            JLabel label = new JLabel(imageIcon);
            frame.getContentPane().add(label);
            frame.pack();
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        }
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
    private static Tensor runAI(Session sess, Tensor inputTensor, String input, String output) throws IOException, FileNotFoundException {
        return sess.runner()
                .feed(input, inputTensor)
                .fetch(output)
                .run().get(0);
    }


    private BufferedImage[] convert(Tensor out) {
        FloatBuffer imageData = FloatBuffer.allocate(num_images * 128 * 128 * 3);
        out.writeTo(imageData);
        imageData.rewind();
        BufferedImage[] images = new BufferedImage[num_images];
        float[] min = new float[num_images];
        float[] max = new float[num_images];
        float[] dif = new float[num_images];
        for (int k = 0; k < num_images; k++) {
            min[k] = Float.MAX_VALUE;
            max[k] = Float.MIN_VALUE;
        }
        for (int k = 0; k < num_images; k++) {
            // fill rgbArray for BufferedImage
            for (int y = WIDTH - 1; y > -1; --y) {
                for (int x = 0; x < WIDTH; ++x) {
                    min[k] = Math.min(min[k], imageData.get());
                    max[k] = Math.max(max[k], imageData.get());
                    dif[k] = max[k] - min[k];
                }
            }
//            System.out.println(k+"\t"+min[k]+"\t"+max[k]+"\t"+dif[k]);
        }

        imageData.rewind();
        for (int k = 0; k < num_images; k++) {
            int r, g, b, i = 0;
            int[] rgbArray = new int[WIDTH * WIDTH];
            for (int y = WIDTH - 1; y > -1; --y) {
                for (int x = 0; x < WIDTH; ++x) {
                    r = (int) ((imageData.get() - min[k]) * 255f / dif[k]) << 16;
                    g = (int) ((imageData.get() - min[k]) * 255f / dif[k]) << 8;
                    b = (int) ((imageData.get() - min[k]) * 255f / dif[k]);
                    i = ((WIDTH - 1) - y) * WIDTH + x;
                    rgbArray[i] = r + g + b;
                }
            }

            // create and save image
            BufferedImage image = new BufferedImage(WIDTH, WIDTH, BufferedImage.TYPE_INT_RGB);
            image.setRGB(0, 0, WIDTH, WIDTH, rgbArray, 0, WIDTH);
            images[k] = image;

        }
        return images;
    }

    public BufferedImage[] scale(BufferedImage[] originals, int size) {
        BufferedImage[] resized = new BufferedImage[num_images];
        for (int k = 0; k < num_images; k++) {
            BufferedImage resize = new BufferedImage(size, size, originals[k].getType());
            Graphics2D g = resize.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(originals[k], 0, 0, size, size, 0, 0, originals[k].getWidth(),
                    originals[k].getHeight(), null);
            g.dispose();
            resized[k] = resize;
        }
        return resized;
    }


}
