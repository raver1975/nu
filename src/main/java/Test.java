import com.idrsolutions.image.JDeli;
import com.idrsolutions.image.scale.SuperResolution;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

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
import java.awt.Label;
import java.awt.Panel;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;

import static org.bytedeco.javacpp.avutil.AV_PIX_FMT_ARGB;


public class Test {

    static final int WIDTH = 128;
    private final int reset_after = 600;
    private static final boolean usesuperresolution = false;
    static int SCAlE_WIDTH = usesuperresolution ? 256 : 256;  //change resolution here
    private static boolean record = false;
    private float speedFactor =20f;

    private Java2DFrameConverter converter;
    public static FFmpegFrameRecorder recorder;
    private int num_images = 1;
    float[][] input = new float[num_images][100];
    float[][] direction = new float[num_images][100];

    public static void main(String[] args) throws IOException, InterruptedException {
        new Test();
    }

    public Test() throws IOException, InterruptedException {
        if (record) {
            Thread thread = new Thread(() -> {
                if (record) {
                    try {
                        System.out.println("recorder stopped");
                        recorder.stop();
                    } catch (FrameRecorder.Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            Runtime.getRuntime().addShutdownHook(thread);
            recorder = new FFmpegFrameRecorder(new File("out" + Math.random() + ".mp4"), SCAlE_WIDTH, SCAlE_WIDTH, 2);
//            recorder.setVideoCodec(12);
            recorder.setFormat("mp4");
            recorder.setFrameRate(30);
            recorder.setVideoQuality(0);
            recorder.setImageWidth(SCAlE_WIDTH);
            recorder.setImageHeight(SCAlE_WIDTH);
            try {
                recorder.start();
                System.out.println("****recorder started");
            } catch (FrameRecorder.Exception e) {
                e.printStackTrace();
            }
            converter = new Java2DFrameConverter();
        }


        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        ImageIcon[] imageIcon = new ImageIcon[num_images];
        JLabel[] label = new JLabel[num_images];
        GridLayout gridLayout = new GridLayout((int) Math.ceil(Math.sqrt(num_images)), (int) Math.ceil(Math.sqrt(num_images)) + 2);
        GridLayout gridLayout1 = new GridLayout(25, 1);
//        GridLayout gridLayout2 = new GridLayout(50, 1);
        frame.getContentPane().setLayout(gridLayout);

        JPanel[] panels = new JPanel[4];
        for (int i = 0; i < 4; i++) {
            panels[i] = new JPanel();
            panels[i].setLayout(gridLayout1);
        }
        JSlider[] sliders = new JSlider[100];
        for (int i = 0; i < 100; i++) {
            sliders[i] = new JSlider();
            sliders[i].setMinimum(-1000);
            sliders[i].setMaximum((1000));
            sliders[i].setName(i + "");
            sliders[i].addChangeListener(new ChangeListener() {

                @Override
                public void stateChanged(ChangeEvent e) {
//                    for (int k = 0; k < 4; k++) {
                    input[0][Integer.parseInt(((JSlider) e.getSource()).getName())] = ((JSlider) e.getSource()).getValue() / 1000f;
//                    }
                    //direction[0][Integer.parseInt(((JSlider) e.getSource()).getName())] = 0f;
                }
            });
            panels[i % 4].add(sliders[i]);
        }
//        frame.getContentPane().add(panels[0]);
        for (int k = 0; k < num_images; k++) {
            imageIcon[k] = new ImageIcon(new BufferedImage(SCAlE_WIDTH, SCAlE_WIDTH, BufferedImage.TYPE_INT_RGB));
            label[k] = new JLabel(imageIcon[k]);
            label[k].addMouseListener(new MouseListener() {
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
         /*   if (k == 2) {
                frame.getContentPane().add(panels[1]);
                frame.getContentPane().add(panels[2]);
            }*/
            frame.getContentPane().add(label[k]);
        }
        //frame.getContentPane().add(panels[3]);

        frame.setUndecorated(false);
        Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
//        frame.setLocation(dim.width / 2 - WIDTH / 2, dim.height / 2 - this.WIDTH / 2);
        frame.pack();
        frame.setVisible(true);
        frame.setSize(frame.getSize().width * 2 / 3, frame.getSize().height * 2 / 3);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // load the model Bundle

        try (SavedModelBundle smb = SavedModelBundle.load("./model", "serve")) {
            Session sess = smb.session();
            Graph graph = smb.graph();
            resetImage();
            int cnt = 0;
            while (true) {
                if (++cnt % reset_after == 0) {
                    resetImage();
                }
                for (int k = 0; k < num_images; k++) {
                    for (int i = 0; i < 100; i++) {
                        input[k][i] += direction[k][i];
                        if (input[k][i] < -1f || input[k][i] > 1f) {
                            direction[k][i] = -direction[k][i];
                            input[k][i] += direction[k][i];
                        }
//                        direction[k][i] += ((float) (Math.random() * 2.0f) - 1f) / 500f;
                    }
                    //normalize(direction,k, speed);

                }
                for (int i = 0; i < 100; i++) {
                    sliders[i].setValue((int) (input[0][i] * 1000f));
                }

                Tensor t = Tensor.create(input);
                Tensor out = runAI(sess, t, "input_z", "generator/out");

//                Tensor discr = runAI(sess, out, "inputs_real", "discriminator/out");
//                FloatBuffer discrData = FloatBuffer.allocate(1);
//                discr.writeTo(discrData);
//                float discAr = discrData.array()[0];
//                System.out.println(discAr);

                BufferedImage[] images;
                images = usesuperresolution ? convert(out) : scale(convert(out), SCAlE_WIDTH);
//                for (BufferedImage image:images) {
//                }

                //twoD.translate(SCAlE_WIDTH/2+SCAlE_WIDTH/3,SCAlE_WIDTH/2-SCAlE_WIDTH/3);
//                drawClock(images[0],SCAlE_WIDTH/2+SCAlE_WIDTH/2,SCAlE_WIDTH/2+SCAlE_WIDTH/2);
//                drawClock(images[2],SCAlE_WIDTH/2+SCAlE_WIDTH/2,SCAlE_WIDTH/2-SCAlE_WIDTH/2);
//                drawClock(images[1],SCAlE_WIDTH/2-SCAlE_WIDTH/2,SCAlE_WIDTH/2+SCAlE_WIDTH/2);
//                drawClock(images[3],SCAlE_WIDTH/2-SCAlE_WIDTH/2,SCAlE_WIDTH/2-SCAlE_WIDTH/2);
                for (int k = 0; k < num_images; k++) {
                    imageIcon[k].setImage(usesuperresolution ? SuperResolution.scale2x(images[k]) : images[k]);
                    label[k].repaint();
                }
                if (record) {
                    Frame frame1 = converter.convert(images[0]);
                    try {
                        recorder.recordImage(frame1.imageWidth, frame1.imageHeight, frame1.imageDepth, frame1.imageChannels, frame1.imageStride, AV_PIX_FMT_ARGB, frame1.image);
                    } catch (FrameRecorder.Exception e) {
                        e.printStackTrace();
                    }
                }
            }

        }
    }

    private void resetImage() {
        for (int i = 0; i < 100; i++) {
            float ran2 = (float) (Math.random() * 2.0f - 1f);
            for (int k = 0; k < num_images; k++) {
                direction[k][i] = (float) ((Math.random() * 2.0f - 1f)) / speedFactor;
                input[k][i] = ran2;
            }
        }
        //normalize(direction,k, speed);


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


    public BufferedImage[] convert(Tensor out) {
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


    public void normalize(float[][] inp, int k, float scale) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (float f : inp[k]) {
            if (f < min) min = f;
            if (f > max) max = f;
        }

        for (int i = 0; i < input[k].length; i++) {
            inp[k][i] = (((inp[k][i] - min) / (max - min)) * 2f - 1f) * scale;
        }
    }

    public void drawTime(BufferedImage image, int x, int y) {
        Date date = new Date();
        int seconds = date.getSeconds();
        int minutes = date.getMinutes();
        int hours = date.getHours();

        Graphics2D twoD = (Graphics2D) image.getGraphics();
        twoD.setColor(new Color(255, 0, 0, 199));
        SimpleDateFormat sdf = new SimpleDateFormat("hh:MM:ss");
        twoD.setColor(Color.red);
        Font font = new Font(Font.SANS_SERIF, Font.PLAIN, 100);
        twoD.setFont(font);
        twoD.drawString(sdf.format(date), x, y);
    }

    public void drawClock(BufferedImage image, int x, int y) {
        Date date = new Date();
        int seconds = date.getSeconds();
        int minutes = date.getMinutes();
        int hours = date.getHours();

        Graphics2D twoD = (Graphics2D) image.getGraphics();
        twoD.setColor(new Color(255, 0, 0, 199));
//        twoD.translate(SCAlE_WIDTH/2+SCAlE_WIDTH/3,SCAlE_WIDTH/2-SCAlE_WIDTH/3);
        twoD.translate(x, y);

//        Drawing the hour markers
//        for(int i=0; i<12; i++) {
//            twoD.drawLine(0, -SCAlE_WIDTH/4, 0, -SCAlE_WIDTH/2);
//            twoD.rotate(Math.PI/6);
//        }
        twoD.rotate(seconds * Math.PI / 30);
        twoD.setStroke(new BasicStroke(1));
        twoD.setColor(new Color(255, 255, 0, 199));
        twoD.drawLine(0, 0, 0, -SCAlE_WIDTH / 6);

        twoD.rotate(2 * Math.PI - seconds * Math.PI / 30);
        twoD.rotate(minutes * Math.PI / 30);
        twoD.setStroke(new BasicStroke(3));
        twoD.setColor(new Color(0, 0, 255, 199));
        twoD.drawLine(0, 0, 0, -SCAlE_WIDTH / 4);

        twoD.rotate(2 * Math.PI - minutes * Math.PI / 30);
        twoD.rotate(hours * Math.PI / 6);
        twoD.setStroke(new BasicStroke(6));
        twoD.setColor(new Color(255, 0, 0, 199));
        twoD.drawLine(0, 0, 0, -SCAlE_WIDTH / 5);
    }
}
