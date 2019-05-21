import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class Udp {

    public static void main(String[] args) throws IOException, InterruptedException {
        new Udp();
    }

    private Udp() throws IOException, InterruptedException {
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        GridLayout gridLayout = new GridLayout(1, 1);
        frame.getContentPane().setLayout(gridLayout);
        ImageIcon imageIcon = new ImageIcon(new BufferedImage(480, 320, BufferedImage.TYPE_INT_RGB));
        JLabel label = new JLabel(imageIcon);
        frame.getContentPane().add(label);
        frame.setUndecorated(true);
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        FFmpegFrameGrabber fg = new FFmpegFrameGrabber("udp://192.168.1.56:2345");
        fg.setFrameRate(30);
        fg.setFormat("mpegts");
        fg.setVideoBitrate(1000000);
        fg.start();
        Java2DFrameConverter converter = new Java2DFrameConverter();
        while (true) {
            Frame fideorame = fg.grabImage();
            imageIcon.setImage(converter.convert(fideorame));
            label.repaint();
        }
    }

}
